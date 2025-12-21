---
tags:
  - resource
  - documentation
  - coe
  - correction-of-error
  - incident-analysis
  - best-practices
keywords:
  - COE documentation
  - incident analysis
  - post-incident review
  - root cause analysis
  - five whys
  - corrective actions
  - lessons learned
topics:
  - incident management
  - documentation standards
  - operational excellence
  - quality assurance
language: markdown
date of note: 2025-12-19
---

# Correction of Error (COE) Documentation Guide

## Purpose

This guide provides a standardized framework for documenting post-incident analyses through Correction of Error (COE) documents. COEs are critical tools for learning from failures, preventing recurrence, and sharing knowledge across teams and organizations.

## What is a COE?

A COE (Correction of Error) is a structured post-incident analysis document that:

1. **Documents what happened** - Provides a clear, factual account of an incident
2. **Analyzes why it happened** - Uses systematic root cause analysis
3. **Prevents recurrence** - Identifies specific, actionable improvements
4. **Shares learnings** - Enables organizational learning and knowledge transfer

## When to Write a COE

Write a COE for:

- Any incident that causes customer impact
- Any incident in a system that has room for improvement
- Any procedural miss that could benefit from analysis
- Any event where learnings could help prevent future issues

**Key Principle**: COEs are improvement opportunities, not punitive measures. If in doubt, write a COE.

## COE Document Structure

### Required Sections

#### 1. Executive Summary

**Purpose**: Provide a concise overview that anyone can understand quickly.

**Content** (1-3 paragraphs):
- **Background**: Brief introduction to the system/service (1-2 sentences)
- **What Happened**: Clear description of the incident
- **Why It Happened**: Root causes at a high level
- **What We're Doing**: Key corrective actions

**Best Practices**:
- Spell out acronyms on first use
- Be specific with times, dates, and timezones
- Use format: "X of Y (Z percent)" for figures
- Avoid customer names
- Keep brief but complete

**Example Structure**:
```markdown
[Service Name] is a [brief description]. On [DATE] between [START] and [END] 
[TIMEZONE], customers experienced [SPECIFIC IMPACT] affecting [NUMBER] 
customers/requests/systems.

The incident was caused by [PRIMARY CAUSE] which led to [CASCADING EFFECTS]. 
Recovery occurred when [RECOVERY ACTION] at [TIME]. It took X minutes to 
detect, Y minutes to mitigate, and Z minutes to fully resolve.

Key lessons and actions focus on: [LESSON 1], [LESSON 2], [LESSON 3].
```

#### 2. Related Documents

**Purpose**: Link to relevant context and related analyses.

**Content**:
- Design documents
- Previous related COEs
- Architectural decision records
- System documentation
- Monitoring dashboards

**Format**:
```markdown
- **[Document Title](./path/to/document.md)** - Brief description
- **[System Architecture](../design/system_design.md)** - Context document
```

#### 3. Timeline of Events

**Purpose**: Chronological account of what happened and when.

**Content**:
- Start with triggering event (may precede customer impact)
- Include customer impact start and end times
- Document detection, escalation, mitigation, and resolution
- Note all operator actions and system events
- Include links to tickets, logs, deployments

**Best Practices**:
- Use consistent timezone throughout
- Use consistent timestamp format (YYYY-MM-DD HH:MM:SS TZ)
- **Bold important milestones**:
  - **START OF CUSTOMER IMPACT**
  - **END OF CUSTOMER IMPACT**
  - Detection time
  - Mitigation time
- Add "Minutes from Impact" calculations if helpful
- Include links to all referenced resources
- Order chronologically

**Format**:
```markdown
**YYYY-MM-DD HH:MM TZ** - [Actor/System] Description of event
  - Additional context
  - Links to resources
```

#### 4. Customer Impact

**Purpose**: Quantify and describe the actual impact on customers/users.

**Content**:
- Start and end times for each affected region/service
- Blast radius metrics:
  - Number of affected customers/users
  - Number of failed requests/operations
  - Percentage of traffic affected
  - Error rates and latencies
- Specific symptoms customers experienced
- What was NOT impacted (important context)
- Screenshots of customer-facing errors if relevant

**Best Practices**:
- Use specific numbers and percentages
- Include units (seconds, requests, customers)
- If you lack instrumentation to measure impact, add an action item
- Consider: "How could you cut blast radius in half?"
- Break down impact by severity/type if multiple impact modes
- DO NOT mention customers by name

**Example**:
```markdown
During the 62 minutes of impact, customers experienced:
- 691,252,856 failed GET requests out of 24,600,749,571 total requests (2.8%)
- Peak latency of 5.2 seconds (P99) vs normal 250ms
- Complete unavailability for 220,000 unique customers
- No impact to existing sessions or data integrity
```

#### 5. Metrics and Graphs

**Purpose**: Visual evidence of impact and system behavior.

**Content**:
- Direct customer impact metrics (errors, latency)
- System health metrics (CPU, memory, connections)
- Comparison with baseline/normal behavior
- Vertical lines marking key timeline events

**Best Practices**:
- Start with most relevant graph (usually customer impact)
- Snapshot images (data may shift over time)
- Also provide links to live metrics/dashboards
- Use consistent scales across related graphs
- Label axes clearly with units and intervals
- Add horizontal lines for thresholds/baselines
- Add vertical lines for key timeline events
- Keep graph sizes reasonable
- Include alt text for accessibility
- Prefer colors with different darkness levels (colorblind friendly)

#### 6. What Went Well

**Purpose**: Highlight practices that reduced impact or aided recovery.

**Content**:
- Tools and processes that helped
- Automation that reduced manual effort
- Effective monitoring or alerting
- Communication that helped
- Load shedding or graceful degradation

**Examples**:
- "Automated rollback reduced recovery time by 30 minutes"
- "Pre-defined runbooks enabled quick diagnosis"
- "Load shedding kept 50% of traffic working vs. 0%"

#### 7. Incident Response Analysis

**Purpose**: Analyze how quickly the team detected and responded to the incident.

**Key Metrics**:
- **Time to Detect**: From impact start to team awareness
- **Time to Diagnose**: From impact start to understanding root cause
- **Time to Mitigate**: From detection to impact resolution
- **Time to Resolve**: From impact start to full resolution (may differ from mitigation)

**Questions to Answer**:
1. How was the event detected? (alarm, customer report, manual)
2. How could detection time be halved?
3. How did you determine how to mitigate?
4. How could mitigation time be halved?

**Mitigation vs. Resolution**:
- **Mitigation**: Customer impact returns to normal (e.g., rollback deployment)
- **Resolution**: Root cause fixed and confidence restored (e.g., bug fixed)
- These may happen at different times

#### 8. Post-Incident Analysis

**Purpose**: Identify opportunities to have prevented the incident.

**Questions to Answer**:
1. How was root cause diagnosed?
2. How could diagnosis time be halved?
3. Was there an existing backlog item that would've prevented this?
   - If yes, why wasn't it completed?
4. Was there an automated check that could've caught this?
5. Was this triggered by a change?
   - Should testing have caught it?
   - Was there proper change management?
6. What specific tool/command triggered impact?
7. Would an operational readiness review have helped?

#### 9. Root Cause Analysis (Five Whys)

**Purpose**: Systematically identify the underlying causes that, if addressed, would prevent similar incidents.

**The Five Whys Method**:
1. Start with the problem (customer impact)
2. Ask "Why?" until you reach actionable root causes
3. Branch your analysis tree (multiple contributing causes)
4. Don't stop at human error - ask why the system allowed it
5. Continue past 5 if needed (5 is minimum, not maximum)
6. Reference action items directly in your analysis

**Structure**:
```markdown
**Why #1: Why did the incident happen?**
Answer: [Direct technical cause]

**Why #2: Why did [cause from #1] occur?**
Answer: [Underlying cause]
[Branch if multiple causes]

**Why #3: Why did [cause from #2] occur?**
Answer: [Deeper cause]
**Action Item**: [Specific action to address this cause]

[Continue until reaching systemic/process causes]
```

**Best Practices**:
- Each "why" builds on the previous answer
- Create branches (1.1, 1.2, 1.2.1) for multiple causes
- Don't stop at "human error" - ask why the error was possible
- Consider separate trees for:
  - Why it happened (root cause)
  - Why it took so long (response)
  - Why detection was slow
- Link each cause to action items
- Be specific and actionable
- Avoid blame - focus on systems and processes

**Anti-Patterns to Avoid**:
- "CPU utilization became excessive" (Why? Keep asking)
- "Service X failed" (Why wasn't your system resilient?)
- "Operator error" (What allowed the operator to make that error?)

#### 10. Lessons Learned

**Purpose**: Distill key takeaways that can be shared broadly.

**Content**:
- 3-10 bulleted insights
- Focus on what's most valuable for others to know
- Link each lesson to specific action items
- Consider what other teams should learn

**Categories**:
- **Technical**: Architecture, design, implementation
- **Process**: Testing, deployment, change management
- **Operational**: Monitoring, alerting, response
- **Cultural**: Ownership, communication, priorities

**Format**:
```markdown
- **Lesson Title**: Explanation of what was learned and why it matters. 
  [ACTION-1] [ACTION-2]
```

#### 11. Action Items

**Purpose**: Specific, measurable steps to prevent recurrence.

**SMART Criteria**:
- **S**pecific: Clear what needs to be done
- **M**easurable: Can verify completion
- **A**ssignable: Has clear owner
- **R**ealistic: Can be completed in timeframe
- **T**ime-bound: Has specific due date

**Priority Levels**:
- **P0 (Critical)**: Root cause of customer impact or extended duration
  - Due: 15-30 days
  - Pre-empts other work
- **P1 (High)**: Lessons learned to improve operations
  - Due: 60-90 days
  - Must be done this quarter
- **P2 (Medium)**: Preventative maintenance for future scale
  - Due: 90-180 days
  - Should be done this year

**Format**:
```markdown
| Action Item | Owner | Priority | Due Date | Status |
|------------|-------|----------|----------|--------|
| [Description with link] | Team/Person | P0 | YYYY-MM-DD | 🔴 Not Started |
```

**Best Practices**:
- Word as user stories with clear exit criteria
- Include verification criteria
- Link to tracking systems (Jira, SIM, etc.)
- For large projects, action is "create plan" with link
- Address classes of problems, not just point fixes
- Avoid far-future dates (max 90 days typical)
- Don't propose unrealistic redesigns
- Maintain status as work progresses

#### 12. Appendices (Optional)

**Purpose**: Provide supporting information without cluttering main narrative.

**Content**:
- Technical background for non-experts
- Detailed metrics or logs
- Additional context or terminology
- Extended technical details

---

## Anti-Patterns to Avoid

### Content Anti-Patterns

❌ **DON'T**:
- Mention customers by name
- Speculate on customer business impact
- Enter legally privileged information
- Blame individuals or teams
- Use "operator error" as a root cause
- Have only "improve documentation" as action items
- Take action items you won't complete
- Use vague or open-ended action items
- Spend unbounded time (complete quickly even if imperfect)

✅ **DO**:
- Focus on systems and processes
- Take ownership ("what could WE have done differently?")
- Ask why operators could make errors
- Have specific, technical action items
- Set realistic timeframes
- Be blameless and forward-looking

### Process Anti-Patterns

❌ **DON'T**:
- Assign COEs punitively or passive-aggressively
- Refuse COEs asked for in good faith
- Skip COEs because "nothing interesting to learn"
- Wait weeks before starting the COE
- Let COEs languish for weeks

✅ **DO**:
- Start immediately after impact is mitigated
- Prioritize COE completion highly
- Complete within 2 weeks (typical)
- View COEs as improvement opportunities
- Involve all relevant teams early

---

## Writing Tips

### Style Guidelines

1. **Be Specific**:
   - ✅ "Between 10:30 and 11:15 PST"
   - ❌ "In the morning"

2. **Include Context**:
   - ✅ "15,000 of 1,000,000 requests failed (1.5%)"
   - ❌ "Many requests failed"

3. **Use Consistent Format**:
   - Timestamps: YYYY-MM-DD HH:MM:SS TZ
   - Metrics: X of Y (Z%)
   - Links: Always include

4. **Be Clear and Direct**:
   - ✅ "The cache failed because..."
   - ❌ "It seems like maybe there was a cache issue..."

5. **Avoid Jargon**:
   - Spell out acronyms on first use
   - Explain technical concepts briefly
   - Consider readers from other teams

### Common Mistakes

1. **Insufficient Detail**: 
   - Problem: "The system had issues"
   - Fix: "The auth service failed 50% of requests between 10:00-10:15 PST"

2. **Missing Links**:
   - Problem: "We deployed version 1.2.3"
   - Fix: "We deployed version 1.2.3 ([CR-123456](link))"

3. **Unclear Metrics**:
   - Problem: "High error rate"
   - Fix: "Error rate peaked at 25% (1,250 of 5,000 requests)"

4. **Timeline Gaps**:
   - Problem: Missing 30-minute period in timeline
   - Fix: Explain gap: "10:30-11:00: Team investigating logs (no actions taken)"

---

## Review Checklist

Before publishing, verify:

### Quality Checks

**Completeness**:
- [ ] All required sections present
- [ ] Timeline is complete and chronological
- [ ] All metrics included with units
- [ ] All questions answered
- [ ] All action items have owners and dates

**Technical Accuracy**:
- [ ] Timeline verified against logs/metrics
- [ ] Impact numbers verified
- [ ] Root cause analysis reaches systemic causes
- [ ] Action items address root causes

**Clarity**:
- [ ] Acronyms spelled out on first use
- [ ] Technical concepts explained
- [ ] Graphs labeled clearly
- [ ] Links provided for all references
- [ ] Written in third person
- [ ] No customer names mentioned

**Sensitivity**:
- [ ] No customer information exposed
- [ ] No security vulnerabilities detailed
- [ ] No confidential project names (use secure COE if needed)
- [ ] No attack vectors described
- [ ] No employee names mentioned (except public escalations)

**Actionability**:
- [ ] Five Whys reaches root causes
- [ ] Each root cause has action items
- [ ] Action items are SMART
- [ ] Action items address classes of problems
- [ ] Lessons learned connect to actions

---

## COE Workflow

### 1. Immediately After Incident

1. Create COE (even if details unclear)
2. Record initial timeline from memory
3. Gather logs, metrics, screenshots
4. Note who was involved

### 2. Within 24 Hours

1. Complete detailed timeline
2. Document customer impact metrics
3. Begin Five Whys analysis
4. Draft action items

### 3. Within 1 Week

1. Complete all sections
2. Review with team
3. Refine root cause analysis
4. Finalize action items
5. Request review from peer/manager

### 4. Within 2 Weeks (Typical Deadline)

1. Address review feedback
2. Get final approval
3. Publish COE
4. Share with relevant teams
5. Begin working on action items

### 5. Follow-up

1. Track action item completion
2. Update COE with learnings from actions
3. Share results with organization

---

## Templates

### Executive Summary Template

```markdown
## Executive Summary

**Incident Type**: [Production Failure / Deployment Issue / etc.]
**Severity**: [SEV-1 / SEV-2 / etc.]
**Duration**: [X hours/minutes] ([START TIME] → [END TIME])
**Impact**: [Brief description of customer impact]

**What Happened**: [Service/System] experienced [specific problem] 
affecting [number] customers/requests. The incident was triggered by 
[immediate cause] which led to [cascading effects].

**Root Causes**:
- [Primary root cause]
- [Secondary root cause]
- [Contributing factors]

**Current Status**: [Resolution status]. [Any remaining work or follow-up needed]

**Key Takeaways**:
- [Critical lesson 1]
- [Critical lesson 2]
- [Critical lesson 3]
```

### Timeline Entry Template

```markdown
**YYYY-MM-DD HH:MM:SS TZ** - [Actor/System] [Action/Event]
  - [Additional context]
  - [Link to ticket/CR/deployment]
  - [Metrics or observations]
```

### Five Whys Template

```markdown
### Root Cause Analysis

**1. Why did [the incident] happen?**
Because [direct technical cause].

**2. Why did [cause from #1] occur?**
Because [underlying cause].

**2.1 Why [alternative cause]?**
Because [branching cause].
**ACTION**: [Specific action to address this]

**3. Why did [cause from #2] occur?**
Because [deeper cause].
**ACTION**: [Specific action to address this]

[Continue until reaching systemic/process causes]
```

### Action Item Template

```markdown
| Priority | Action Item | Owner | Due Date | Status |
|----------|------------|-------|----------|--------|
| P0 | [Implement circuit breaker pattern in Auth service](link-to-ticket) | Backend Team | 2025-12-31 | 🔴 Not Started |
| P1 | [Add automated testing for deployment rollback](link-to-ticket) | DevOps Team | 2025-01-15 | 🟡 In Progress |
| P2 | [Document incident response runbook](link-to-ticket) | SRE Team | 2025-02-01 | 🟢 Complete |
```

---

## Examples of Effective COEs

### Strong Executive Summaries

**Example 1** (Specific, Clear, Complete):
```markdown
On 2025-12-15 between 10:30 and 11:45 PST (75 minutes), S3 customers 
experienced a 2.8% error rate for GET requests in the IAD region. 
691,252,856 GET failures occurred out of 24,600,749,571 total requests.

The incident was caused by elevated memory pressure in the Keymap 
Functional Coordinator (KFC) caching fleet following a cold restart of 
multiple hosts. Recovery occurred when we rolled back the deployment 
and increased heap size.

Key lessons: (1) Cold restart procedures need testing at scale, 
(2) Deployment automation lacked adequate safeguards, (3) Metrics 
weren't sufficient to quickly diagnose heap pressure issues.
```

**Example 2** (Multi-Service Impact):
```markdown
On 2025-12-19 between 14:51 and 17:15 PST, Amazon Chime experienced 
complete unavailability for meetings and messaging. 27,817 calls failed 
(60% of expected traffic) affecting 220,000 unique customers.

The messaging service Profile table was throttled by DynamoDB after 
the Hazelcast cache failed due to file descriptor exhaustion. Meetings 
depend on messaging for chat rooms, creating cascading failure.

Key lessons: (1) Service dependencies created blast radius amplification, 
(2) Load testing didn't simulate cache failure, (3) File descriptor limits 
weren't monitored.
```

### Strong Root Cause Analysis

**Example** (Multi-Level Analysis):
```markdown
**1. Why was there customer impact?**
Because the authentication service returned 500 errors for 50% of requests.

**2. Why did auth service return errors?**
Because the database connection pool was exhausted (0 available connections).

**2.1 Why was the connection pool exhausted?**
Because query latency increased from 10ms to 5000ms (500x increase).
**ACTION**: Add connection pool monitoring and alerting

**2.2 Why were queries slow?**
Because a missing index on the users table caused full table scans.

**3. Why was the index missing?**
Because our migration script failed silently during deployment.

**3.1 Why did the migration fail silently?**
Because we don't validate schema changes before promoting to production.
**ACTION**: Add schema validation to deployment pipeline

**3.2 Why don't we validate schemas?**
Because this isn't part of our standard deployment process.
**ACTION**: Update deployment checklist to require schema validation

**4. Why didn't our testing catch this?**
Because our test database is small (1000 rows) and queries complete quickly 
even with full table scans. Production has 10M rows.
**ACTION**: Create production-scale test environment
```

---

## FAQ

### When should I write a COE?

- Any customer-impacting incident
- Any system failure with improvement opportunities
- Any procedural miss worth analyzing
- When multiple services are involved (each may write separate COEs)

### How long should a COE take?

- **Target**: Complete within 2 weeks of incident
- **Start**: Immediately after impact is mitigated
- **Priority**: Make this highest priority work
- **Effort**: Don't spend unbounded time - complete even if imperfect

### What if multiple teams are involved?

- Each team should write their own COE
- Cross-reference related COEs
- Collaborate on shared understanding
- Own your team's improvements

### What if I don't have all the metrics?

- Document what you know
- Add action item to implement missing instrumentation
- Use approximate numbers with clear caveats
- Focus on what you can measure

### Should I make the COE public?

- Default: Yes, share broadly for organizational learning
- Secure: Only for confidential projects or security issues
- Team-only: Only for very low/zero impact with minimal learnings

### What if the root cause is complex?

- Break down into smaller causal chains
- Use multiple Five Whys branches
- Focus on most actionable causes
- It's okay to have multiple root causes

### What makes a good action item?

- Specific and testable
- Has clear owner
- Has realistic due date
- Addresses root cause, not symptoms
- Solves class of problems, not just one instance

---

## Summary

A well-written COE:

1. **Documents clearly** what happened and when
2. **Quantifies impact** with specific metrics
3. **Analyzes deeply** using Five Whys to find root causes
4. **Proposes solutions** through specific, actionable items
5. **Shares learnings** so others can benefit
6. **Takes ownership** rather than assigning blame
7. **Completes quickly** to maintain context and momentum

The goal is not perfection but learning and improvement. A COE is a living document that helps teams and organizations grow stronger through systematic analysis of failures.

---

## Additional Resources

- Amazon COE User Guide: w.amazon.com/bin/view/NewCOE/UserGuide
- Five Whys Method: w.amazon.com/bin/view/NewCOE/UserGuide#HMoreonrootcausing
- Example COEs: See "Examples of great COEs" section in User Guide
- COE Bar Raiser Program: For review and quality standards
- Policy Engine: For automated checks and policy compliance

---

*This guide is based on Amazon's COE standards and practices. Teams should adapt it to their specific needs while maintaining core principles of blameless analysis and actionable improvement.*

*Last Updated: 2025-12-19*  
*Maintainer: ML Platform Team*
