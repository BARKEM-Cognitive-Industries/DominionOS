//! Generic content-addressed store (`ContentStore<V>`).
//!
//! Several modules in dominion-core independently maintained a
//! `BTreeMap<Hash256, V>` with matching publish / fetch / contains / remove
//! helpers. This module provides one canonical implementation so callers can
//! delegate to it instead of reimplementing the pattern.
//!
//! ## The `ContentAddressed` trait
//!
//! A value type participates in the store by implementing [`ContentAddressed`],
//! which returns the SHA-256 content id for that value.  Both [`Object`] (via
//! `obj.id()`) and [`Package`] (via `pkg.content_id`) already produce a
//! [`Hash256`]; the trait just names that operation uniformly so the store can
//! call it without knowing the concrete type.
//!
//! [`Object`]: crate::object::Object
//! [`Package`]: crate::packaging::Package

use crate::hash::Hash256;
use alloc::collections::BTreeMap;

// ── trait ─────────────────────────────────────────────────────────────────────

/// A value that knows its own content address.
///
/// Implement this for any type whose content id is a `Hash256` derived from its
/// bytes — either computed on the fly (like `Object::id()`) or stored as a
/// pre-computed field (like `Package::content_id`).
pub trait ContentAddressed {
    fn content_id(&self) -> Hash256;
}

// ── ContentStore ──────────────────────────────────────────────────────────────

/// A content-addressed map from [`Hash256`] to `V`.
///
/// `publish` derives the key from the value itself (via [`ContentAddressed`])
/// so callers never supply a key independently — the store is always
/// self-consistent.  Identical values deduplicate to a single entry.
pub struct ContentStore<V> {
    entries: BTreeMap<Hash256, V>,
}

impl<V: ContentAddressed + Clone> ContentStore<V> {
    /// Create an empty store.
    pub fn new() -> Self {
        ContentStore { entries: BTreeMap::new() }
    }

    /// Insert `content` into the store and return its content id.
    ///
    /// If an entry with the same id already exists (identical content), the
    /// existing entry is kept unchanged and the id is returned as-is (dedup).
    pub fn publish(&mut self, content: V) -> Hash256 {
        let id = content.content_id();
        self.entries.entry(id).or_insert(content);
        id
    }

    /// Look up a value by its content id.
    pub fn fetch(&self, id: &Hash256) -> Option<&V> {
        self.entries.get(id)
    }

    /// Return `true` iff the store holds an entry for `id`.
    pub fn contains(&self, id: &Hash256) -> bool {
        self.entries.contains_key(id)
    }

    /// Remove and return the entry for `id`, or `None` if absent.
    pub fn remove(&mut self, id: &Hash256) -> Option<V> {
        self.entries.remove(id)
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` iff the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate all `(id, value)` pairs in ascending id order.
    pub fn iter(&self) -> impl Iterator<Item = (&Hash256, &V)> {
        self.entries.iter()
    }
}

impl<V: ContentAddressed + Clone> Default for ContentStore<V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── trait impls for existing content-addressed types ─────────────────────────

impl ContentAddressed for crate::object::Object {
    /// SHA-256 of the canonical `obj1` encoding — matches `Object::id()`.
    fn content_id(&self) -> Hash256 {
        self.id()
    }
}

impl ContentAddressed for crate::packaging::Package {
    /// The pre-computed `content_id` stored in the package at seal time.
    fn content_id(&self) -> Hash256 {
        self.content_id
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{Datum, Object};

    fn doc(body: &str) -> Object {
        Object::new("Doc").with("body", Datum::Text(body.into()))
    }

    #[test]
    fn publish_returns_content_id() {
        let mut store: ContentStore<Object> = ContentStore::new();
        let obj = doc("hello");
        let expected = obj.id();
        let id = store.publish(obj);
        assert_eq!(id, expected);
    }

    #[test]
    fn fetch_round_trips() {
        let mut store: ContentStore<Object> = ContentStore::new();
        let id = store.publish(doc("world"));
        let got = store.fetch(&id).unwrap();
        assert_eq!(got.kind, "Doc");
    }

    #[test]
    fn identical_values_deduplicate() {
        let mut store: ContentStore<Object> = ContentStore::new();
        let id1 = store.publish(doc("same"));
        let id2 = store.publish(doc("same"));
        assert_eq!(id1, id2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn contains_and_remove() {
        let mut store: ContentStore<Object> = ContentStore::new();
        let id = store.publish(doc("bye"));
        assert!(store.contains(&id));
        let v = store.remove(&id);
        assert!(v.is_some());
        assert!(!store.contains(&id));
        assert!(store.is_empty());
    }

    #[test]
    fn iter_yields_all_entries() {
        let mut store: ContentStore<Object> = ContentStore::new();
        store.publish(doc("a"));
        store.publish(doc("b"));
        store.publish(doc("c"));
        assert_eq!(store.iter().count(), 3);
    }
}
