//! URL parsing and relative-reference resolution — the addressing layer beneath the
//! browser engine ([`crate::webengine`]).
//!
//! A [`Url`] is split into scheme, host, port, path, query and fragment. The parser
//! handles the schemes the OS cares about — `http`, `https`, and the native
//! `dominion`/`ndn` names — and [`Url::join`] resolves a relative reference against a
//! base (RFC 3986 §5, the common cases: absolute URLs, scheme-relative `//host`,
//! absolute paths `/p`, relative paths `p`, query-only `?q`, fragment-only `#f`, and
//! `.`/`..` path segment normalisation). Pure, safe, host-tested.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A parsed URL.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Url {
    pub scheme: String,
    /// Host (empty for opaque/native names that keep everything in `path`).
    pub host: String,
    /// Explicit port if present, else the scheme default via [`Url::port`].
    pub explicit_port: Option<u16>,
    /// Path, always beginning with `/` for hierarchical (http) URLs.
    pub path: String,
    /// Query string without the leading `?`.
    pub query: Option<String>,
    /// Fragment without the leading `#`.
    pub fragment: Option<String>,
}

impl Url {
    /// Is this one of the OS's native, content-addressed schemes?
    pub fn is_native(&self) -> bool {
        self.scheme == "dominion" || self.scheme == "ndn"
    }

    /// Does the scheme require TLS?
    pub fn is_secure(&self) -> bool {
        self.scheme == "https"
    }

    /// The effective port: explicit if given, else the scheme default.
    pub fn port(&self) -> u16 {
        self.explicit_port.unwrap_or(match self.scheme.as_str() {
            "https" => 443,
            "http" => 80,
            _ => 0,
        })
    }

    /// Parse an absolute URL. A bare string with no scheme is treated as an
    /// `http://` host (so typing `example.com` works), unless it looks like a
    /// native name (`dominion://…`/`ndn:…`).
    pub fn parse(input: &str) -> Option<Url> {
        let s = input.trim();
        if s.is_empty() {
            return None;
        }

        // Split scheme. A scheme is letters/digits/+/-/. followed by ':'.
        let (scheme, rest) = match split_scheme(s) {
            Some((sc, r)) => (sc, r),
            None => {
                // No scheme: default to http for hierarchical names.
                return Url::parse(&("http://".to_string() + s));
            }
        };

        // Native, opaque schemes: keep the remainder (minus an optional `//`) as the
        // host+path so `dominion://home` and `ndn:/jayden/page` both work.
        if scheme == "dominion" || scheme == "ndn" {
            let rest = rest.strip_prefix("//").unwrap_or(rest);
            let (before_frag, fragment) = split_off(rest, '#');
            let (before_query, query) = split_off(before_frag, '?');
            // For native names, host is the first segment, path the rest — but the
            // whole name is what the resolver keys on, so keep host empty and path full.
            return Some(Url {
                scheme,
                host: String::new(),
                explicit_port: None,
                path: before_query.to_string(),
                query: query.map(str::to_string),
                fragment: fragment.map(str::to_string),
            });
        }

        // Hierarchical (http/https): require the `//` authority introducer.
        let after = rest.strip_prefix("//").unwrap_or(rest);

        // authority is up to the first '/', '?', or '#'.
        let auth_end = after.find(['/', '?', '#']).unwrap_or(after.len());
        let authority = &after[..auth_end];
        let remainder = &after[auth_end..];

        // Strip userinfo@ if present.
        let hostport = authority.rsplit('@').next().unwrap_or(authority);
        let (host, explicit_port) = parse_hostport(hostport)?;
        if host.is_empty() {
            return None;
        }

        let (path_q, fragment) = split_off(remainder, '#');
        let (path, query) = split_off(path_q, '?');
        let path = if path.is_empty() { "/".to_string() } else { path.to_string() };

        Some(Url {
            scheme,
            host,
            explicit_port,
            path,
            query: query.map(str::to_string),
            fragment: fragment.map(str::to_string),
        })
    }

    /// Resolve a (possibly relative) reference against this base URL (RFC 3986 §5.2,
    /// the cases a browser hits: absolute, `//authority`, `/abs/path`, `rel/path`,
    /// `?query`, `#frag`, with `.`/`..` normalised).
    pub fn join(&self, reference: &str) -> Option<Url> {
        let r = reference.trim();
        if r.is_empty() {
            return Some(self.clone());
        }
        // Absolute reference with its own scheme.
        if split_scheme(r).is_some() && !r.starts_with("//") {
            // But "scheme:" only counts if it's a known/again-parseable absolute URL.
            if let Some(u) = Url::parse(r) {
                // Guard: `parse` defaults bare names to http; only treat as absolute if
                // the reference actually carried a scheme.
                if r.starts_with(&(u.scheme.clone() + ":")) {
                    return Some(u);
                }
            }
        }
        let mut out = self.clone();
        out.fragment = None;

        if let Some(rest) = r.strip_prefix("//") {
            // Scheme-relative: keep our scheme, replace authority+path.
            let reparsed = Url::parse(&(self.scheme.clone() + "://" + rest))?;
            return Some(reparsed);
        }
        if let Some(frag) = r.strip_prefix('#') {
            out.fragment = Some(frag.to_string());
            return Some(out);
        }
        if let Some(q) = r.strip_prefix('?') {
            let (q, frag) = split_off(q, '#');
            out.query = Some(q.to_string());
            out.fragment = frag.map(str::to_string);
            return Some(out);
        }

        // Path reference. Split its own query/fragment off first.
        let (path_part, frag) = split_off(r, '#');
        let (path_part, query) = split_off(path_part, '?');

        let merged = if let Some(abs) = path_part.strip_prefix('/') {
            // Absolute path.
            "/".to_string() + abs
        } else {
            // Relative path: replace the last segment of the base path.
            let base = &self.path;
            let cut = base.rfind('/').map(|i| i + 1).unwrap_or(0);
            base[..cut].to_string() + path_part
        };
        out.path = normalise_path(&merged);
        out.query = query.map(str::to_string);
        out.fragment = frag.map(str::to_string);
        Some(out)
    }

    /// The `host[:port]` authority for connecting.
    pub fn authority(&self) -> String {
        match self.explicit_port {
            Some(p) => {
                let mut s = self.host.clone();
                s.push(':');
                push_u16(&mut s, p);
                s
            }
            None => self.host.clone(),
        }
    }

    /// The request-target a server expects in the HTTP request line (`path[?query]`).
    pub fn request_target(&self) -> String {
        let mut s = if self.path.is_empty() { "/".to_string() } else { self.path.clone() };
        if let Some(q) = &self.query {
            s.push('?');
            s.push_str(q);
        }
        s
    }

    /// Reassemble the full URL as text.
    pub fn to_string_full(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.scheme);
        if self.is_native() {
            s.push_str("://");
            s.push_str(&self.path);
        } else {
            s.push_str("://");
            s.push_str(&self.host);
            if let Some(p) = self.explicit_port {
                s.push(':');
                push_u16(&mut s, p);
            }
            s.push_str(&self.path);
        }
        if let Some(q) = &self.query {
            s.push('?');
            s.push_str(q);
        }
        if let Some(f) = &self.fragment {
            s.push('#');
            s.push_str(f);
        }
        s
    }
}

/// Split a leading `scheme:` if the prefix is a valid scheme. Returns
/// `(lowercased scheme, remainder after the colon)`.
fn split_scheme(s: &str) -> Option<(String, &str)> {
    let colon = s.find(':')?;
    if colon == 0 {
        return None;
    }
    let scheme = &s[..colon];
    let mut chars = scheme.chars();
    let first = chars.next()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    if !scheme.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.') {
        return None;
    }
    Some((scheme.to_ascii_lowercase(), &s[colon + 1..]))
}

/// Split `s` at the first occurrence of `sep`: `(before, Some(after))` or
/// `(s, None)` if absent.
fn split_off(s: &str, sep: char) -> (&str, Option<&str>) {
    match s.find(sep) {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    }
}

/// Parse `host[:port]`. Lowercases the host. Returns None on a malformed port.
fn parse_hostport(s: &str) -> Option<(String, Option<u16>)> {
    // IPv6 literals would be in [..]; the OS doesn't need them yet, so a plain split.
    match s.rfind(':') {
        Some(i) if s[i + 1..].chars().all(|c| c.is_ascii_digit()) && i + 1 < s.len() => {
            let port: u16 = parse_u16(&s[i + 1..])?;
            Some((s[..i].to_ascii_lowercase(), Some(port)))
        }
        _ => Some((s.to_ascii_lowercase(), None)),
    }
}

/// Remove `.`/`..` segments from a path (RFC 3986 §5.2.4).
fn normalise_path(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let absolute = path.starts_with('/');
    let trailing = path.ends_with('/') && path.len() > 1;
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    let mut s = String::new();
    if absolute {
        s.push('/');
    }
    s.push_str(&out.join("/"));
    if trailing && !s.ends_with('/') {
        s.push('/');
    }
    if s.is_empty() {
        s.push('/');
    }
    s
}

fn parse_u16(s: &str) -> Option<u16> {
    let mut n: u32 = 0;
    for c in s.chars() {
        let d = c.to_digit(10)?;
        n = n.checked_mul(10)?.checked_add(d)?;
        if n > u16::MAX as u32 {
            return None;
        }
    }
    Some(n as u16)
}

fn push_u16(s: &mut String, n: u16) {
    if n >= 10 {
        push_u16(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_http_url() {
        let u = Url::parse("http://example.com/a/b?x=1#top").unwrap();
        assert_eq!(u.scheme, "http");
        assert_eq!(u.host, "example.com");
        assert_eq!(u.port(), 80);
        assert_eq!(u.path, "/a/b");
        assert_eq!(u.query.as_deref(), Some("x=1"));
        assert_eq!(u.fragment.as_deref(), Some("top"));
    }

    #[test]
    fn https_default_port_and_explicit_port() {
        assert_eq!(Url::parse("https://a.com").unwrap().port(), 443);
        assert!(Url::parse("https://a.com").unwrap().is_secure());
        let u = Url::parse("http://a.com:8080/x").unwrap();
        assert_eq!(u.explicit_port, Some(8080));
        assert_eq!(u.authority(), "a.com:8080");
    }

    #[test]
    fn bare_host_defaults_to_http() {
        let u = Url::parse("example.com/path").unwrap();
        assert_eq!(u.scheme, "http");
        assert_eq!(u.host, "example.com");
        assert_eq!(u.path, "/path");
    }

    #[test]
    fn empty_path_becomes_root() {
        let u = Url::parse("http://example.com").unwrap();
        assert_eq!(u.path, "/");
        assert_eq!(u.request_target(), "/");
    }

    #[test]
    fn native_schemes_are_opaque() {
        let a = Url::parse("dominion://home").unwrap();
        assert!(a.is_native());
        assert_eq!(a.scheme, "dominion");
        assert_eq!(a.path, "home");
        let n = Url::parse("ndn:/jayden/page").unwrap();
        assert!(n.is_native());
        assert_eq!(n.path, "/jayden/page");
    }

    #[test]
    fn userinfo_is_stripped() {
        let u = Url::parse("http://user:pw@host.com/x").unwrap();
        assert_eq!(u.host, "host.com");
    }

    #[test]
    fn join_absolute_reference_wins() {
        let base = Url::parse("http://a.com/dir/page").unwrap();
        let u = base.join("https://b.com/other").unwrap();
        assert_eq!(u.host, "b.com");
        assert_eq!(u.scheme, "https");
    }

    #[test]
    fn join_absolute_path_replaces_path() {
        let base = Url::parse("http://a.com/dir/page?q=1").unwrap();
        let u = base.join("/new/path").unwrap();
        assert_eq!(u.host, "a.com");
        assert_eq!(u.path, "/new/path");
        assert_eq!(u.query, None);
    }

    #[test]
    fn join_relative_path_replaces_last_segment() {
        let base = Url::parse("http://a.com/dir/page").unwrap();
        let u = base.join("sub").unwrap();
        assert_eq!(u.path, "/dir/sub");
    }

    #[test]
    fn join_dotdot_normalises() {
        let base = Url::parse("http://a.com/dir/sub/page").unwrap();
        let u = base.join("../up").unwrap();
        assert_eq!(u.path, "/dir/up");
    }

    #[test]
    fn join_query_and_fragment_only() {
        let base = Url::parse("http://a.com/p?old#x").unwrap();
        assert_eq!(base.join("?new").unwrap().query.as_deref(), Some("new"));
        assert_eq!(base.join("#sec").unwrap().fragment.as_deref(), Some("sec"));
        // Fragment-only keeps the path+query.
        let f = base.join("#sec").unwrap();
        assert_eq!(f.path, "/p");
        assert_eq!(f.query.as_deref(), Some("old"));
    }

    #[test]
    fn join_scheme_relative() {
        let base = Url::parse("https://a.com/p").unwrap();
        let u = base.join("//cdn.com/lib.js").unwrap();
        assert_eq!(u.scheme, "https");
        assert_eq!(u.host, "cdn.com");
        assert_eq!(u.path, "/lib.js");
    }

    #[test]
    fn round_trips_to_string() {
        let u = Url::parse("http://example.com:8080/a/b?x=1#top").unwrap();
        assert_eq!(u.to_string_full(), "http://example.com:8080/a/b?x=1#top");
    }
}
