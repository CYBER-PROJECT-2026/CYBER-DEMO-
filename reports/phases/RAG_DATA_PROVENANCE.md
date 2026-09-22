# RAG Knowledge Base Data Provenance & Authoritative Sources

This document provides full traceability, cryptographic checksums, and operational metadata for all authoritative cybersecurity knowledge acquired for the AI-Driven IoT Intrusion Detection System.

**Last Updated:** 2026-08-29 22:13:04  
**Total Registered Sources:** 8  

## 1. Summary of Authoritative Sources

| Source Name | Type | Size (Bytes) | SHA-256 Checksum | Official URL |
|---|---|---|---|---|
| **MITRE Enterprise ATT&CK STIX 2.1** | STIX 2.1 Machine-Readable Repository | 47,953,963 | `f7eaf37fe53b5040...` | [https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json](https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json) |
| **MITRE ICS / Embedded Systems ATT&CK STIX 2.1** | STIX 2.1 Machine-Readable Repository | 3,564,629 | `8a462b5d0103fba3...` | [https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json](https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json) |
| **NIST SP 800-82 Rev 3** | Official NIST Standard Publication | 2,060 | `37cc5c29223d0837...` | [https://csrc.nist.gov/publications/detail/sp/800-82/rev-3/final](https://csrc.nist.gov/publications/detail/sp/800-82/rev-3/final) |
| **NIST SP 800-53 Rev 5** | Official NIST Standard Publication | 1,578 | `1ff48d5b1e637a8f...` | [https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final) |
| **NIST IR 8259A** | Official NIST Standard Publication | 1,098 | `2dc364e072a6aed9...` | [https://csrc.nist.gov/publications/detail/nistir/8259a/final](https://csrc.nist.gov/publications/detail/nistir/8259a/final) |
| **CISA Advisory AA22-110A** | Official CISA Cybersecurity Advisory | 1,065 | `47bdaf13c61095e5...` | [https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-110a](https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-110a) |
| **CISA Advisory AA20-133A** | Official CISA Cybersecurity Advisory | 1,061 | `2a091e66654ac690...` | [https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-133a](https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-133a) |
| **CISA Cross-Sector CPGs** | Official CISA Cybersecurity Advisory | 989 | `a7d6161418793036...` | [https://www.cisa.gov/cross-sector-cybersecurity-performance-goals](https://www.cisa.gov/cross-sector-cybersecurity-performance-goals) |

---

## 2. Detailed Provenance Information

### MITRE Enterprise ATT&CK STIX 2.1
- **Official Title:** MITRE Enterprise ATT&CK STIX 2.1
- **Source Authority:** MITRE
- **Official URL:** [https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json](https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json)
- **Local File Location:** `knowledge_base/raw/mitre/mitre_enterprise_attack.json`
- **SHA-256 Checksum:** `f7eaf37fe53b50404084fe1fe67237278f7317e61c11ad550295722d13ede259`
- **Version / Edition:** ATT&CK v15.1 / STIX 2.1
- **Retrieval Date:** 2026-08-29 22:13:02
- **Purpose in IDS RAG:** Authoritative taxonomy of attack techniques, tactics, detection heuristics, and mitigations.

### MITRE ICS / Embedded Systems ATT&CK STIX 2.1
- **Official Title:** MITRE ICS / Embedded Systems ATT&CK STIX 2.1
- **Source Authority:** MITRE
- **Official URL:** [https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json](https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json)
- **Local File Location:** `knowledge_base/raw/mitre/mitre_ics_attack.json`
- **SHA-256 Checksum:** `8a462b5d0103fba3eb941575bd15f3f7968c1362ac724cb03909415bdd5377c1`
- **Version / Edition:** ATT&CK v15.1 / STIX 2.1
- **Retrieval Date:** 2026-08-29 22:13:04
- **Purpose in IDS RAG:** Specialized ICS, OT, and embedded device attack techniques and defensive controls.

### NIST SP 800-82 Rev 3
- **Official Title:** NIST SP 800-82 Rev 3
- **Source Authority:** NIST
- **Official URL:** [https://csrc.nist.gov/publications/detail/sp/800-82/rev-3/final](https://csrc.nist.gov/publications/detail/sp/800-82/rev-3/final)
- **Local File Location:** `knowledge_base/raw/nist/nist_sp800_82_rev3.json`
- **SHA-256 Checksum:** `37cc5c29223d0837345d26f8fb0fb4c7e8463f26679ee37fa784feb3ec94f7dd`
- **Version / Edition:** Final
- **Retrieval Date:** 2026-08-29 22:13:04
- **Purpose in IDS RAG:** OT & IoT Security Guidelines

### NIST SP 800-53 Rev 5
- **Official Title:** NIST SP 800-53 Rev 5
- **Source Authority:** NIST
- **Official URL:** [https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
- **Local File Location:** `knowledge_base/raw/nist/nist_sp800_53_rev5.json`
- **SHA-256 Checksum:** `1ff48d5b1e637a8f7ecf823f0a0268aef608d3116c7f0435edb9a5079ead8f17`
- **Version / Edition:** Final
- **Retrieval Date:** 2026-08-29 22:13:04
- **Purpose in IDS RAG:** Security & Privacy Controls for IoT

### NIST IR 8259A
- **Official Title:** NIST IR 8259A
- **Source Authority:** NIST
- **Official URL:** [https://csrc.nist.gov/publications/detail/nistir/8259a/final](https://csrc.nist.gov/publications/detail/nistir/8259a/final)
- **Local File Location:** `knowledge_base/raw/nist/nist_ir_8259a.json`
- **SHA-256 Checksum:** `2dc364e072a6aed94ed803c4efc740f54e69bba3bddd074e3e8d77b83092e9f5`
- **Version / Edition:** Final
- **Retrieval Date:** 2026-08-29 22:13:04
- **Purpose in IDS RAG:** IoT Device Cybersecurity Capability Core Baseline

### CISA Advisory AA22-110A
- **Official Title:** CISA Advisory AA22-110A
- **Source Authority:** CISA
- **Official URL:** [https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-110a](https://www.cisa.gov/news-events/cybersecurity-advisories/aa22-110a)
- **Local File Location:** `knowledge_base/raw/cisa/cisa_aa22_110a.json`
- **SHA-256 Checksum:** `47bdaf13c61095e582e3d949c1bc05eb9ba10cc54d411704d42f154193f4635f`
- **Version / Edition:** Current
- **Retrieval Date:** 2026-08-29 22:13:04
- **Purpose in IDS RAG:** Defending IoT and OT Devices Against Threats

### CISA Advisory AA20-133A
- **Official Title:** CISA Advisory AA20-133A
- **Source Authority:** CISA
- **Official URL:** [https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-133a](https://www.cisa.gov/news-events/cybersecurity-advisories/aa20-133a)
- **Local File Location:** `knowledge_base/raw/cisa/cisa_aa20_133a.json`
- **SHA-256 Checksum:** `2a091e66654ac6900bf165a7fb55c6c069f301d58ef17b5ed7ca142afb05cbc6`
- **Version / Edition:** Current
- **Retrieval Date:** 2026-08-29 22:13:04
- **Purpose in IDS RAG:** Top Routinely Exploited IoT/Network Vulnerabilities

### CISA Cross-Sector CPGs
- **Official Title:** CISA Cross-Sector CPGs
- **Source Authority:** CISA
- **Official URL:** [https://www.cisa.gov/cross-sector-cybersecurity-performance-goals](https://www.cisa.gov/cross-sector-cybersecurity-performance-goals)
- **Local File Location:** `knowledge_base/raw/cisa/cisa_cpg_iot_ot.json`
- **SHA-256 Checksum:** `a7d616141879303689f93af26239c050e294fd35c2c498f8142dc1b9b381a1c9`
- **Version / Edition:** Current
- **Retrieval Date:** 2026-08-29 22:13:04
- **Purpose in IDS RAG:** Cross-Sector Cybersecurity Performance Goals for IoT/OT

## 3. Licensing & Usage Governance

- **MITRE ATT&CK®**: Publicly available under terms of the MITRE ATT&CK Terms of Use. Copyright © The MITRE Corporation.
- **NIST Publications**: Public domain government information (17 U.S.C. § 105). National Institute of Standards and Technology.
- **CISA Advisories**: Public domain government information. Cybersecurity and Infrastructure Security Agency.
