//! Anonymous credentials & **selective disclosure** — closes the AJ/AI partial
//! ("full attribute/predicate credential schema deferred"). Builds on the existing ZK and
//! anonymity substrate ([`crate::zkservice`] membership, [`crate::anon`] unlinkable pseudonyms,
//! [`crate::zk`] Schnorr) with a concrete credential schema:
//!
//! * An **issuer** attests a set of attributes by committing them in a Merkle tree and signing
//!   the root (a hash-based, PQ-safe signature via [`crate::bft::Signer`] — one published issuer
//!   key, many credentials).
//! * The **holder** presents only the attributes a verifier needs ([`Credential::present`]),
//!   proving each is genuinely in the signed credential by Merkle path — **without revealing the
//!   others**. Birthdate stays hidden; an issuer-attested `over_18 = true` predicate attribute is
//!   what's shown.
//! * Presentations are **unlinkable**: each carries a per-context pseudonym derived from the
//!   holder's secret + the context, so two presentations of the same credential to two verifiers
//!   cannot be correlated (the [`crate::anon`] property).
//!
//! Pure, safe `no_std`, host- and metal-tested.

use crate::bft::{verify_sig, OtsSig, Signer};
use crate::hash::Hash256;
use alloc::string::String;
use alloc::vec::Vec;

fn merkle_node(a: &Hash256, b: &Hash256) -> Hash256 {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(&a.0);
    input[32..].copy_from_slice(&b.0);
    Hash256::of(&input)
}

/// One attested attribute (name → value), e.g. `("over_18", "true")` or `("country", "AU")`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attribute {
    pub name: String,
    pub value: Vec<u8>,
}

impl Attribute {
    pub fn new(name: &str, value: &[u8]) -> Attribute {
        Attribute { name: String::from(name), value: value.to_vec() }
    }
    /// The Merkle leaf for this attribute (binds name and value).
    fn leaf(&self) -> Hash256 {
        let mut input = Vec::with_capacity(self.name.len() + self.value.len() + 8);
        input.extend_from_slice(b"attr:");
        input.extend_from_slice(self.name.as_bytes());
        input.push(b'=');
        input.extend_from_slice(&self.value);
        Hash256::of(&input)
    }
}

/// A Merkle tree over an ordered attribute list (padded to a power of two by repeating leaves).
struct Tree {
    // `levels[0]` is the leaf row, so the leaves need no separate field.
    levels: Vec<Vec<Hash256>>,
}

impl Tree {
    fn build(attrs: &[Attribute]) -> Tree {
        let mut leaves: Vec<Hash256> = attrs.iter().map(|a| a.leaf()).collect();
        if leaves.is_empty() {
            leaves.push(Hash256::ZERO);
        }
        while !leaves.len().is_power_of_two() {
            leaves.push(*leaves.last().unwrap());
        }
        let mut levels = alloc::vec![leaves.clone()];
        let mut level = leaves.clone();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len() / 2);
            for pair in level.chunks(2) {
                next.push(merkle_node(&pair[0], &pair[1]));
            }
            levels.push(next.clone());
            level = next;
        }
        Tree { levels }
    }
    fn root(&self) -> Hash256 {
        *self.levels.last().and_then(|l| l.first()).unwrap_or(&Hash256::ZERO)
    }
    fn path(&self, mut idx: usize) -> Vec<Hash256> {
        let mut path = Vec::new();
        for level in &self.levels {
            if level.len() == 1 {
                break;
            }
            let sib = idx ^ 1;
            path.push(level.get(sib).copied().unwrap_or(level[idx]));
            idx /= 2;
        }
        path
    }
}

fn verify_path(leaf: Hash256, mut idx: usize, path: &[Hash256], root: Hash256) -> bool {
    let mut acc = leaf;
    for sib in path {
        acc = if idx & 1 == 0 { merkle_node(&acc, sib) } else { merkle_node(sib, &acc) };
        idx /= 2;
    }
    acc == root
}

/// A credential held by a user: the full attribute set (private), the issuer's signed root, and
/// the holder's secret (for unlinkable pseudonyms). Never transmitted whole.
pub struct Credential {
    attributes: Vec<Attribute>,
    root: Hash256,
    issuer_sig: OtsSig,
    holder_secret: [u8; 32],
}

/// A disclosed attribute plus its Merkle proof.
#[derive(Clone, Debug)]
pub struct DisclosedAttribute {
    pub attribute: Attribute,
    index: usize,
    path: Vec<Hash256>,
}

/// What a holder hands a verifier: only the chosen attributes (with proofs), the issuer
/// signature over the credential root, and a per-context unlinkable pseudonym.
pub struct Presentation {
    pub root: Hash256,
    issuer_sig: OtsSig,
    disclosed: Vec<DisclosedAttribute>,
    pub pseudonym: Hash256,
}

impl Credential {
    /// Issue a credential: the issuer commits `attributes` and signs the root with the next
    /// available one-time key slot under its published [`Signer`] root.
    ///
    /// Consumes one OTS slot from `issuer`. Returns `None` when the issuer's key set is
    /// exhausted — the issuer must rotate to a new [`Signer`] (fresh Merkle key set) before
    /// issuing further credentials. Accepting a caller-supplied index would allow a caller to
    /// reuse index 0 for two different credential roots, leaking enough Lamport preimage bits
    /// to forge arbitrary signatures under the issuer's published key.
    pub fn issue(issuer: &mut Signer, attributes: Vec<Attribute>, holder_secret: &[u8]) -> Option<Credential> {
        let tree = Tree::build(&attributes);
        let root = tree.root();
        let issuer_sig = issuer.sign_next(&root.0)?;
        Some(Credential { attributes, root, issuer_sig, holder_secret: Hash256::of(holder_secret).0 })
    }

    pub fn attribute_names(&self) -> Vec<&str> {
        self.attributes.iter().map(|a| a.name.as_str()).collect()
    }

    /// The unlinkable pseudonym this holder presents to `context` (per-verifier, uncorrelatable).
    fn pseudonym_for(&self, context: &str) -> Hash256 {
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(b"AE-NYM");
        input.extend_from_slice(&self.holder_secret);
        input.extend_from_slice(context.as_bytes());
        Hash256::of(&input)
    }

    /// Present only the attributes named in `disclose` to `context`. Undisclosed attributes are
    /// not included — the verifier never sees them.
    pub fn present(&self, disclose: &[&str], context: &str) -> Presentation {
        let tree = Tree::build(&self.attributes);
        let mut out = Vec::new();
        for (i, a) in self.attributes.iter().enumerate() {
            if disclose.contains(&a.name.as_str()) {
                out.push(DisclosedAttribute { attribute: a.clone(), index: i, path: tree.path(i) });
            }
        }
        Presentation {
            root: self.root,
            issuer_sig: self.issuer_sig.clone(),
            disclosed: out,
            pseudonym: self.pseudonym_for(context),
        }
    }
}

impl Presentation {
    /// The disclosed attributes a verifier sees.
    pub fn disclosed(&self) -> Vec<&Attribute> {
        self.disclosed.iter().map(|d| &d.attribute).collect()
    }
}

/// Verify a presentation against the issuer's published key: the issuer signed the root, and
/// every disclosed attribute proves into that root. Returns the verified attributes on success.
pub fn verify_presentation(issuer_root: Hash256, p: &Presentation) -> Option<Vec<Attribute>> {
    // 1. The issuer genuinely signed this credential root.
    if !verify_sig(issuer_root, &p.root.0, &p.issuer_sig) {
        return None;
    }
    // 2. Every disclosed attribute is a real leaf of the signed root.
    let mut verified = Vec::new();
    for d in &p.disclosed {
        if !verify_path(d.attribute.leaf(), d.index, &d.path, p.root) {
            return None;
        }
        verified.push(d.attribute.clone());
    }
    Some(verified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bft::Signer;

    fn issuer() -> Signer {
        Signer::new(b"gov-issuer-key", 4)
    }

    fn sample(issuer: &mut Signer) -> Credential {
        let attrs = alloc::vec![
            Attribute::new("name", b"Jayden"),
            Attribute::new("birthdate", b"1990-01-01"),
            Attribute::new("over_18", b"true"),
            Attribute::new("country", b"AU"),
        ];
        Credential::issue(issuer, attrs, b"holder-secret").unwrap()
    }

    #[test]
    fn selective_disclosure_reveals_only_chosen_attributes() {
        let mut iss = issuer();
        let cred = sample(&mut iss);
        // Disclose only the predicate, not the birthdate.
        let pres = cred.present(&["over_18"], "bar.example");
        let verified = verify_presentation(iss.public_key(), &pres).unwrap();
        assert_eq!(verified.len(), 1);
        assert_eq!(verified[0].name, "over_18");
        assert_eq!(verified[0].value, b"true");
        // The birthdate is not in the presentation at all.
        assert!(!pres.disclosed().iter().any(|a| a.name == "birthdate"));
    }

    #[test]
    fn a_forged_attribute_fails_verification() {
        let mut iss = issuer();
        let cred = sample(&mut iss);
        let mut pres = cred.present(&["country"], "ctx");
        // Tamper with the disclosed value — its Merkle path no longer proves into the root.
        pres.disclosed[0].attribute.value = b"US".to_vec();
        assert!(verify_presentation(iss.public_key(), &pres).is_none());
    }

    #[test]
    fn a_different_issuer_key_is_rejected() {
        let mut iss = issuer();
        let cred = sample(&mut iss);
        let pres = cred.present(&["over_18"], "ctx");
        let attacker = Signer::new(b"attacker", 4);
        assert!(verify_presentation(attacker.public_key(), &pres).is_none());
        assert!(verify_presentation(iss.public_key(), &pres).is_some());
    }

    #[test]
    fn presentations_are_unlinkable_across_contexts() {
        let mut iss = issuer();
        let cred = sample(&mut iss);
        let p1 = cred.present(&["over_18"], "site-A");
        let p2 = cred.present(&["over_18"], "site-B");
        // Same credential, two verifiers → different pseudonyms (uncorrelatable).
        assert_ne!(p1.pseudonym, p2.pseudonym);
        // But the same context is stable for the holder.
        let p1b = cred.present(&["over_18"], "site-A");
        assert_eq!(p1.pseudonym, p1b.pseudonym);
    }

    #[test]
    fn multiple_attributes_disclose_together() {
        let mut iss = issuer();
        let cred = sample(&mut iss);
        let pres = cred.present(&["over_18", "country"], "ctx");
        let verified = verify_presentation(iss.public_key(), &pres).unwrap();
        assert_eq!(verified.len(), 2);
    }
}
