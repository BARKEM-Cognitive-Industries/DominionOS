# Infrastructure Setup for DominionOS

This guide covers setting up the infrastructure needed for DominionOS to function as a distributed system: the DominionLink network, package repository, and public pool.

---

## Overview

DominionOS is designed to work both standalone (in QEMU or on bare metal) and as part of a distributed network. To fully enable that vision, several services need to be deployed and configured.

**Current Status:** Designed and partially implemented. Not yet in production.

---

## What We Need to Set Up

### 1. DominionLink Network (Named Data Networking)

**What it is:** A decentralized content-addressed network layer. Nodes publish and resolve data by name/hash, similar to IPFS but with capability-based access control.

**What's implemented:**
- ✅ NDN protocol (interest/data packets)
- ✅ Content addressing (SHA-256 hashes)
- ✅ Kademlia DHT (distributed hash table)
- ✅ DominionLink (self-certifying IDs + key-based routing)
- ✅ DNS bridge (DominionLink names ↔ DNS A records)

**What we need to set up:**

#### A. Public DominionLink Bootstrap Nodes
```
Number needed: 3-5 for redundancy
Hardware per node: 2 vCPU, 4 GB RAM, 100 GB SSD (for DHT state)
Location: Geographic diversity (US, EU, Asia preferred)

Role: 
- Accept connections from peers (DominionOS instances)
- Maintain DHT routing table
- Respond to content lookups
- Cache frequently accessed objects
```

**How to deploy:**
```bash
# Build the DominionLink node daemon
cd kernel
cargo build --release --bin dominionlink-node

# Run as a service (systemd example)
[Unit]
Description=DominionLink Bootstrap Node
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/dominionlink-node \
  --listen 0.0.0.0:5000 \
  --dht-table /var/lib/dominionlink/dht.db \
  --cache-size 50G
Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

**Configuration needed:**
- Bootstrap node IDs (hardcoded in DominionOS)
- Network addresses (IPs or DNS names)
- Firewall rules (UDP port 5000, TCP ports optional)
- Monitoring (health checks, uptime tracking)

#### B. DNS Bridge
```
Role: Translate domain names to DominionLink content hashes

Example:
  dominion://my-app/v1.0  → DNS lookup my-app.dominion.link
  → CNAME to dominionlink-bootstrap-1.cognitive-industries.org
  → Server resolves to content hash
  → Client fetches from DHT
```

**How to set up:**
```
1. Register domain: dominion.link (or similar)
2. Set up DNS server (Route53, Cloudflare, or self-hosted)
3. Create CNAME entries for bootstrap nodes
4. Optionally: HTTP gateway at dominion.link for web browsers
```

---

### 2. Package Repository

**What it is:** A registry of pre-built packages (libraries, apps, drivers) that can be installed into DominionOS instances.

**What's implemented:**
- ✅ Package versioning (semantic)
- ✅ Dependency resolution
- ✅ Content addressing (packages as immutable objects)
- ✅ Capability-gated downloads (signed manifests)

**What we need to set up:**

#### A. Central Package Repository Server
```
Role: Store and serve packages

Hardware: 4 vCPU, 8 GB RAM, 1-10 TB SSD (depends on package volume)
Location: Any region (replicated via DominionLink DHT)

Stores:
- Canonical package versions
- Cryptographic signatures
- Release metadata (changelogs, compatibility)
- Package statistics (downloads, usage)
```

**How to deploy:**
```bash
# Build the package server
cd kernel
cargo build --release --bin pkg-server

# Run as a service
[Unit]
Description=DominionOS Package Server
After=network.target dominionlink-node.service

[Service]
Type=simple
ExecStart=/usr/local/bin/pkg-server \
  --listen 0.0.0.0:6000 \
  --dominionlink-bootstrap <bootstrap-node-ids> \
  --storage /var/lib/pkgrepo
Restart=always

[Install]
WantedBy=multi-user.target
```

**Configuration needed:**
- Publishing credentials (who can upload packages?)
- Signing keys (for package authenticity)
- Retention policy (how long do we keep packages?)
- Bandwidth limits (rate limiting, quotas)

#### B. Package Registry (Metadata Service)
```
Role: Catalog all available packages, versions, and metadata

Interface: HTTP API
  GET /api/packages
  GET /api/packages/{name}
  GET /api/packages/{name}/{version}
  POST /api/publish (admin only)
```

**How to set up:**
```
1. Database (PostgreSQL): Store package metadata
2. HTTP server (Actix or Rocket): Serve API
3. Authentication: Signing keys for trusted publishers
4. Replication: Copy to multiple regions
```

Example structure:
```json
{
  "name": "my-library",
  "versions": [
    {
      "version": "1.0.0",
      "dominion-os-min": "1.0.0",
      "content-hash": "sha256:...",
      "size": 102400,
      "published": "2026-06-22T00:00:00Z",
      "signature": "...",
      "dependencies": [
        {"name": "dep-lib", "version": "^2.0"}
      ]
    }
  ]
}
```

---

### 3. Public Compute Pool

**What it is:** A pool of machines offering distributed computing resources (similar to AWS Lambda or Google Cloud Functions, but decentralized).

**What's implemented:**
- ✅ Capability-based job submission
- ✅ Resource accounting (CPU time, memory, I/O)
- ✅ Deterministic execution (reproducible results)
- ✅ Settlement (payment/compensation for compute)

**What we need to set up:**

#### A. Compute Pool Coordinator
```
Role: Accept job submissions, allocate to worker nodes, aggregate results

Hardware: 2 vCPU, 4 GB RAM (lightweight)
Location: Any region

Stores: Active jobs, worker status, results cache
```

**How to deploy:**
```bash
# Build the pool coordinator
cargo build --release --bin pool-coordinator

# Run as a service
[Unit]
Description=DominionOS Compute Pool Coordinator
After=network.target dominionlink-node.service

[Service]
Type=simple
ExecStart=/usr/local/bin/pool-coordinator \
  --listen 0.0.0.0:7000 \
  --dominionlink-bootstrap <bootstrap-node-ids> \
  --min-workers 5
Restart=always

[Install]
WantedBy=multi-user.target
```

#### B. Compute Pool Workers
```
Role: Execute jobs, report results

Hardware: Variable (1+ vCPU, 2+ GB RAM per worker)
Location: Distributed (personal computers, data centers, edge nodes)

Runs: DominionOS in a VM, connects to coordinator, accepts jobs
```

**How to set up:**
```
1. Each organization/individual runs worker nodes (volunteers or commercial)
2. Workers register with coordinator (publish capability)
3. Coordinator sends jobs to workers
4. Workers execute deterministically, return results
5. Coordinator verifies and rewards workers
```

---

## Implementation Checklist

### Phase 1: Minimal Public Network (Week 1-2)

- [ ] **Deploy 3 bootstrap nodes**
  - Hardware: Rent on AWS, DigitalOcean, or equivalent
  - Setup: Install DominionLink node daemon
  - Networking: Open ports, configure DNS
  - Monitoring: Health checks, uptime tracking

- [ ] **Register dominion.link domain**
  - DNS provider: Cloudflare, Route53, or self-hosted
  - CNAME entries: Point to bootstrap nodes
  - DNSSEC: Optional but recommended

- [ ] **Deploy package repository**
  - Server: One central repository (replicated via DHT)
  - Database: PostgreSQL with backups
  - Signing key: Generate and secure-store root key

- [ ] **Documentation**
  - How to connect DominionOS instance to network
  - How to publish packages
  - How to submit jobs (if pool is enabled)

### Phase 2: Resilience & Scale (Week 3-4)

- [ ] **Replicate bootstrap nodes**
  - 5+ geographic locations
  - Automated failover
  - Latency monitoring

- [ ] **Package repository replication**
  - CDN or multi-region copies
  - Background sync
  - Bandwidth optimization

- [ ] **Compute pool pilot**
  - Deploy 5-10 worker nodes
  - Test job submission → execution → results
  - Verify determinism & reproducibility

- [ ] **Monitoring & alerting**
  - Node health dashboards
  - Automated alerts for outages
  - Performance metrics

### Phase 3: Community & Incentives (Week 5+)

- [ ] **Public worker registration**
  - Website for volunteers to register nodes
  - Automated incentive model (rewards/payments)
  - Reputation tracking

- [ ] **Public package publishing**
  - Anyone can publish packages (with community review)
  - Signing & verification infrastructure
  - Package quality metrics

- [ ] **Analytics & governance**
  - Network statistics (active nodes, packages, jobs)
  - Community guidelines
  - Dispute resolution

---

## Cost Estimates

### Infrastructure Costs (Monthly)

```
Bootstrap nodes (3):           $15-30/month
  - 2 vCPU, 4 GB RAM, 100 GB SSD x 3

Package repository:            $10-20/month
  - 2 vCPU, 8 GB RAM, 1 TB SSD

Coordinator:                   $5-10/month
  - 2 vCPU, 4 GB RAM, 100 GB SSD

Domain:                        $12-20/year
  - dominion.link registration

DNS:                           Free-10/month
  - If self-hosted: minimal

CDN (if needed):               $10-50/month
  - For package distribution

Total (minimal):               ~$50-100/month
Total (with CDN):              ~$80-200/month
```

### Personnel Costs

```
Setup (one-time):              40-80 hours
  - Infrastructure provisioning
  - Security hardening
  - Documentation

Maintenance (ongoing):         5-10 hours/week
  - Monitoring
  - Updates
  - User support
```

---

## Security Considerations

### Key Management
```
Root signing key: 
  - Generate offline (air-gapped machine)
  - Store in hardware security module (HSM) or paper backup
  - Never share

Bootstrap node IDs:
  - Public (hardcoded in DominionOS)
  - But: network traffic is encrypted (DominionLink provides encryption)

Package signatures:
  - Each package signed with root key
  - Verification before install (mandatory in DominionOS)
```

### Access Control
```
Package publishing:
  - Restricted to trusted contributors (whitelisted accounts)
  - Eventually: community voting for new publishers

Job submission:
  - Capability-based: only authorized users submit jobs
  - Rate limiting: prevent DoS

Bootstrap node access:
  - Public, but: traffic is rate-limited
  - DDoS protection (optional: Cloudflare, etc.)
```

### Monitoring
```
Node health:
  - Automated checks every 5 minutes
  - Alert if any node down >5 minutes

Package integrity:
  - Periodic verification of signatures
  - Alert if tampering detected

Job reproducibility:
  - Sample jobs re-run on different workers
  - Alert if results diverge (indicates bug or tampering)
```

---

## Current Implementation Status

### What's Built (in dominion-core)
- NDN forwarding (ndn.rs)
- DominionLink (dominionlink.rs, self-certifying IDs)
- Package versioning (packaging.rs)
- Job submission API (marketplace.rs)
- Deterministic execution framework (state.rs)

### What's Not Yet Built (Phase 2 work)
- Bootstrap node daemon (needs wrapping)
- Package repository server (HTTP API)
- Pool coordinator service
- CLI tools for package publishing
- Dashboard for monitoring

### What's Specified But Not Coded (Phase 3 work)
- Community package registry UI
- Worker node incentive model
- Reputation system
- Governance framework

---

## Configuration Files

### Bootstrap Node List
**File:** `dominion-core/config/bootstrap.toml`

```toml
[[bootstrap_nodes]]
name = "bootstrap-1"
node_id = "sha256:..."
addresses = [
  "bootstrap-1.dominion.link:5000",
  "1.2.3.4:5000"
]
dht_port = 5000
https = false

[[bootstrap_nodes]]
name = "bootstrap-2"
node_id = "sha256:..."
addresses = [
  "bootstrap-2.dominion.link:5000",
  "5.6.7.8:5000"
]
```

### Package Repository Config
**File:** `kernel/config/packages.toml`

```toml
[repository]
url = "https://pkgrepo.cognitive-industries.org"
root_signing_key = "sha256:..."
verify_signatures = true

[sync]
enabled = true
interval_hours = 24
max_package_size_mb = 1000
```

---

## Next Steps

1. **Decision:** Do we deploy phase 1 infrastructure now?
2. **Estimate:** How much budget and time?
3. **Assign:** Who owns each component?
4. **Timeline:** When should each phase ship?
5. **Incentives:** How do we reward node operators and contributors?

---

## Questions?

- **Technical details:** See dominion-core source (marketplace.rs, packaging.rs, ndn.rs)
- **Architecture:** See docs/architecture.md → Networking & Distributed Compute sections
- **Setup help:** contact@cognitive-industries.org

---

**Ready to build the network?**

contact@cognitive-industries.org
