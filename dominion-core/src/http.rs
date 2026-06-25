//! A minimal but real HTTP/1.1 client codec — request serialisation and response
//! parsing — sitting above the TCP transport and below the browser engine
//! ([`crate::webengine`]).
//!
//! Serialises a `GET` (or any method) request with the headers a server expects, and
//! parses a response into status line + headers + body, decoding both framing modes a
//! real server uses: `Content-Length` and `Transfer-Encoding: chunked`. Header lookup
//! is case-insensitive (RFC 7230). Redirects are surfaced via [`Response::location`].
//! Pure bytes-in/bytes-out, host-tested — the byte pipe (TCP/TLS) is supplied by the
//! transport.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The product token sent as `User-Agent`.
pub const USER_AGENT: &str = "DominionOS/1.0 (DominionOS; universal-browser)";

/// An HTTP request to serialise onto a connection.
#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    /// The request-target (path?query), from [`crate::url::Url::request_target`].
    pub target: String,
    /// The `Host` header value (authority).
    pub host: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// A `GET` for `target` on `host` with browser-appropriate default headers.
    pub fn get(host: &str, target: &str) -> Request {
        Request {
            method: "GET".to_string(),
            target: target.to_string(),
            host: host.to_string(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }

    pub fn header(mut self, name: &str, value: &str) -> Request {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    /// Serialise to wire bytes. Adds `Host`, `User-Agent`, `Accept`, and
    /// `Connection: close` (we read until EOF, the simplest correct framing) unless
    /// the caller already set them.
    pub fn serialize(&self) -> Vec<u8> {
        // Reserve up front so the header block grows in one allocation rather than
        // a chain of reallocs (each realloc copies and can trigger heap growth):
        // request line + default headers + caller headers + body, plus slack.
        let mut cap = self.method.len() + self.target.len() + self.host.len() + 64;
        for (k, v) in &self.headers {
            cap += k.len() + v.len() + 4;
        }
        cap += self.body.len();
        let mut s = String::with_capacity(cap);
        s.push_str(&self.method);
        s.push(' ');
        s.push_str(if self.target.is_empty() { "/" } else { &self.target });
        s.push_str(" HTTP/1.1\r\n");

        let has = |n: &str| self.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(n));
        if !has("host") {
            s.push_str("Host: ");
            s.push_str(&self.host);
            s.push_str("\r\n");
        }
        if !has("user-agent") {
            s.push_str("User-Agent: ");
            s.push_str(USER_AGENT);
            s.push_str("\r\n");
        }
        if !has("accept") {
            s.push_str("Accept: text/html,*/*\r\n");
        }
        if !has("connection") {
            s.push_str("Connection: close\r\n");
        }
        for (k, v) in &self.headers {
            s.push_str(k);
            s.push_str(": ");
            s.push_str(v);
            s.push_str("\r\n");
        }
        if !self.body.is_empty() && !has("content-length") {
            s.push_str("Content-Length: ");
            push_usize(&mut s, self.body.len());
            s.push_str("\r\n");
        }
        s.push_str("\r\n");
        let mut out = s.into_bytes();
        out.extend_from_slice(&self.body);
        out
    }
}

/// A parsed HTTP response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    /// Case-insensitive header lookup (first match).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The `Content-Type` value (without parameters), lowercased.
    pub fn content_type(&self) -> Option<String> {
        self.header("content-type").map(|ct| {
            ct.split(';').next().unwrap_or(ct).trim().to_ascii_lowercase()
        })
    }

    /// True when the body should be treated as HTML (or no type was given — servers
    /// often omit it for HTML).
    pub fn is_html(&self) -> bool {
        match self.content_type() {
            Some(ct) => ct == "text/html" || ct == "application/xhtml+xml",
            None => true,
        }
    }

    pub fn is_redirect(&self) -> bool {
        matches!(self.status, 301 | 302 | 303 | 307 | 308) && self.location().is_some()
    }

    pub fn location(&self) -> Option<&str> {
        self.header("location")
    }

    /// The body decoded as UTF-8 (lossy — invalid bytes become U+FFFD), for the
    /// HTML/text renderer.
    pub fn body_text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    /// Parse a complete response message from raw bytes (headers + already-collected
    /// body bytes). Decodes `Transfer-Encoding: chunked`; otherwise honours
    /// `Content-Length`, falling back to "the rest of the buffer" (Connection: close
    /// framing).
    pub fn parse(raw: &[u8]) -> Option<Response> {
        let split = find_header_end(raw)?;
        let head = &raw[..split.0];
        let body_bytes = &raw[split.1..];

        let head_str = core::str::from_utf8(head).ok()?;
        let mut lines = head_str.split("\r\n");
        let status_line = lines.next()?;
        let (status, reason) = parse_status_line(status_line)?;

        let mut headers: Vec<(String, String)> = Vec::new();
        for line in lines {
            if line.is_empty() {
                continue;
            }
            if let Some(colon) = line.find(':') {
                let name = line[..colon].trim().to_string();
                let value = line[colon + 1..].trim().to_string();
                headers.push((name, value));
            }
        }

        let chunked = headers
            .iter()
            .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked"));

        let body = if chunked {
            decode_chunked(body_bytes)
        } else if let Some(len) = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| parse_usize(v.trim()))
        {
            body_bytes[..len.min(body_bytes.len())].to_vec()
        } else {
            body_bytes.to_vec()
        };

        Some(Response { status, reason, headers, body })
    }
}

/// Locate the CRLFCRLF (or LFLF) terminating the header block. Returns
/// `(end_of_headers, start_of_body)`.
fn find_header_end(raw: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < raw.len() {
        if raw[i] == b'\r' && raw[i + 1] == b'\n' {
            if i + 3 < raw.len() && raw[i + 2] == b'\r' && raw[i + 3] == b'\n' {
                return Some((i, i + 4));
            }
            if i + 2 < raw.len() && raw[i + 2] == b'\n' {
                // Tolerate a bare LF on the blank line.
                return Some((i, i + 3));
            }
        }
        if raw[i] == b'\n' && raw[i + 1] == b'\n' {
            return Some((i, i + 2));
        }
        i += 1;
    }
    None
}

fn parse_status_line(line: &str) -> Option<(u16, String)> {
    // "HTTP/1.1 200 OK"
    let mut parts = line.splitn(3, ' ');
    let version = parts.next()?;
    if !version.starts_with("HTTP/") {
        return None;
    }
    let code = parts.next()?;
    let status = parse_usize(code)? as u16;
    let reason = parts.next().unwrap_or("").to_string();
    Some((status, reason))
}

/// Decode a `Transfer-Encoding: chunked` body: a sequence of `<hex-len>CRLF<data>CRLF`
/// terminated by a zero-length chunk. Trailers (if any) are ignored.
fn decode_chunked(mut data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // Read the chunk-size line each iteration; stop when no CRLF remains.
    while let Some(nl) = find_crlf(data) {
        let size_line = &data[..nl];
        // Size may carry `;chunk-extensions` — take hex up to ';'.
        let hex_end = size_line.iter().position(|&b| b == b';').unwrap_or(size_line.len());
        let size = match parse_hex(&size_line[..hex_end]) {
            Some(n) => n,
            None => break,
        };
        data = &data[nl + 2..];
        if size == 0 {
            break;
        }
        if size > data.len() {
            // Truncated — take what we have.
            out.extend_from_slice(data);
            break;
        }
        out.extend_from_slice(&data[..size]);
        data = &data[size..];
        // Skip the trailing CRLF after the chunk data.
        if data.len() >= 2 && data[0] == b'\r' && data[1] == b'\n' {
            data = &data[2..];
        } else if !data.is_empty() && data[0] == b'\n' {
            data = &data[1..];
        }
    }
    out
}

fn find_crlf(data: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 1 < data.len() {
        if data[i] == b'\r' && data[i + 1] == b'\n' {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn parse_hex(s: &[u8]) -> Option<usize> {
    let t = core::str::from_utf8(s).ok()?.trim();
    if t.is_empty() {
        return None;
    }
    let mut n: usize = 0;
    for c in t.chars() {
        let d = c.to_digit(16)?;
        n = n.checked_mul(16)?.checked_add(d as usize)?;
    }
    Some(n)
}

fn parse_usize(s: &str) -> Option<usize> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let mut n: usize = 0;
    for c in t.chars() {
        let d = c.to_digit(10)?;
        n = n.checked_mul(10)?.checked_add(d as usize)?;
    }
    Some(n)
}

fn push_usize(s: &mut String, n: usize) {
    if n >= 10 {
        push_usize(s, n / 10);
    }
    s.push((b'0' + (n % 10) as u8) as char);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_a_get_with_default_headers() {
        let req = Request::get("example.com", "/index.html").serialize();
        let s = String::from_utf8(req).unwrap();
        assert!(s.starts_with("GET /index.html HTTP/1.1\r\n"));
        assert!(s.contains("Host: example.com\r\n"));
        assert!(s.contains("User-Agent: DominionOS"));
        assert!(s.contains("Connection: close\r\n"));
        assert!(s.ends_with("\r\n\r\n"));
    }

    #[test]
    fn empty_target_becomes_root() {
        let s = String::from_utf8(Request::get("a.com", "").serialize()).unwrap();
        assert!(s.starts_with("GET / HTTP/1.1"));
    }

    #[test]
    fn parses_content_length_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 5\r\n\r\nhello, world";
        let r = Response::parse(raw).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.reason, "OK");
        assert_eq!(r.body, b"hello");
        assert_eq!(r.content_type().as_deref(), Some("text/html"));
        assert!(r.is_html());
    }

    #[test]
    fn case_insensitive_header_lookup() {
        let raw = b"HTTP/1.1 200 OK\r\nCONTENT-type: Text/HTML; charset=utf-8\r\nContent-Length: 0\r\n\r\n";
        let r = Response::parse(raw).unwrap();
        assert_eq!(r.header("content-type"), Some("Text/HTML; charset=utf-8"));
        assert_eq!(r.content_type().as_deref(), Some("text/html"));
    }

    #[test]
    fn decodes_chunked_body() {
        // "Wiki" + "pedia" in two chunks, then a zero chunk.
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
        let r = Response::parse(raw).unwrap();
        assert_eq!(r.body, b"Wikipedia");
    }

    #[test]
    fn detects_redirects() {
        let raw = b"HTTP/1.1 301 Moved Permanently\r\nLocation: https://example.com/\r\nContent-Length: 0\r\n\r\n";
        let r = Response::parse(raw).unwrap();
        assert!(r.is_redirect());
        assert_eq!(r.location(), Some("https://example.com/"));
    }

    #[test]
    fn missing_content_length_takes_rest() {
        let raw = b"HTTP/1.1 200 OK\r\n\r\nrest of the body until eof";
        let r = Response::parse(raw).unwrap();
        assert_eq!(r.body, b"rest of the body until eof");
    }

    #[test]
    fn body_text_is_lossy_utf8() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi";
        let r = Response::parse(raw).unwrap();
        assert_eq!(r.body_text(), "hi");
    }
}
