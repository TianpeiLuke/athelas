---
tags:
  - resource
  - best-practices
  - coe
  - correction-of-error
  - incident-analysis
  - amazon-practices
keywords:
  - COE best practices
  - incident response
  - root cause analysis
  - five whys
  - action items
  - lessons learned
  - blameless culture
topics:
  - incident management
  - operational excellence
  - quality assurance
  - continuous improvement
  - postmortem analysis
language: markdown
date of note: 2025-12-19
---

# Amazon COE Best Practices

## Purpose

This document summarizes best practices for Correction of Error (COE) process based on industry-leading incident analysis methodologies. These practices complement the [COE Documentation Guide](./coe_documentation_guide.md) with specific standards and cultural expectations for writing high-quality post-incident analyses.

## Core Philosophy

### What COE Is

The **Correction of Errors** process seeks to improve overall system quality by:
1. Finding root causes through deep dive analysis
2. Addressing root causes via trackable action items
3. Stopping re-occurrence of problems
4. Analyzing impact on business and customers
5. Capturing learnings and sharing across Amazon

### What COE Is NOT

1. **NOT** a process for finding blame
2. **NOT** a process for punishing employees
3. **NOT** a punitive mechanism

> "Regardless of what we discover, we understand and truly believe that everyone did the best job they could, given what they knew at the time, their skills and abilities, the resources available, and the situation at hand." 
> — Norm Kerth, Project Retrospectives: A Handbook for Team Review

## When to Write a COE

### Mandatory
- **ALL SEV-1 incidents** (with very few exceptions)
- Customer-impacting incidents
- Incidents affecting multiple services/teams

### Recommended
- SEV-2 incidents with customer impact
- Discovery of unknown single points of failure
- Issues affecting multiple departments
- Any incident with valuable learnings

**Key Principle**: If in doubt, write a COE. COEs are improvement opportunities, not punitive measures.

## Timeline and Ownership

### Completion Timeline
- **Draft**: As soon as possible after incident (information fresh)
- **Target**: Complete within 14 days
- **Review**: Minimum 24 hours in Review status
- **Priority**: Focus on getting information right before word-smithing

### Ownership
- **Owner**: Manager of group responsible for addressing root cause
- **Delegation**: Manager may delegate writing but remains responsible for content
- **Multiple Teams**: Each team should write their own COE (collaboration OK, but own your destiny)

## Content Best Practices

### Executive Summary
- Provide context on service/system and mission
- Include who, when, where, and how of impact
- Include discovery and resolution timelines
- Should stand alone without referencing other sections
- Write as if it will travel as email to VP

### Customer Impact
- **Be Specific**: Use precise numbers, not vague terms
  - ✅ "691,252,856 failed requests out of 24,600,749,571 (2.8%)"
  - ❌ "significant impact" or "some percentage"
- Include Order Impact (even if zero)
- Include UI screenshots of customer experience
- **DO NOT** mention customers by name
- **DO NOT** speculate on customer business impact

### Metrics and Graphs
- **Always** include image snapshots (data may change)
- **Also** include links to live dashboards
- Annotate key points on x-axis
- Use vertical lines for outage start/end times
- Use horizontal lines for SLA breaches
- Label axes with units (seconds, hours, etc.)
- Include sampling intervals (e.g., "Per 10 Minutes")
- Keep graph size reasonable (e.g., 600x400)
- Start with most relevant graph (customer impact)
- Limit to 2-4 powerful graphs that tell the story

### Timeline
- **Be Consistent**: Use same timezone throughout (prefer UTC or "PT")
- **Bold** important milestones (START/END OF IMPACT, detection, mitigation)
- Start with first trigger (not just when team paged)
- Explain gaps >10-15 minutes
- Each entry should flow logically from previous
- Include links to tickets, deployments, logs

### 5 Whys Analysis

#### Core Principles
- **Keep asking** until you have actionable root causes
- **No limit**: 5 is minimum, not maximum
- **Branch**: Multiple root causes = multiple branches
- **Each Why** builds on previous answer
- **Each root cause** → one or more actions with deadline and owner

#### What to Keep Asking
- **DON'T stop at**:
  - Human error (ask why error was possible)
  - Missing procedure (ask how to improve/automate)
  - "Service X failed" (ask why not more resilient)
  - "CPU utilization high" (ask why CPU high)
  - "Host ran out of memory" (ask why ran out)

- **DO address**:
  - Root cause: Why did issue occur?
  - Discovery: Why did it take so long to discover?
  - Resolution: Why did it take so long to resolve?

#### Example Structure
```
Why #1: Why did incident happen?
Answer: [Direct technical cause]

Why #2: Why did [cause from #1] occur?
Answer: [Underlying cause]
  Branch 2.1: [Alternative cause]
  Branch 2.2: [Another cause]

Why #3: Why did [cause from #2] occur?
Answer: [Deeper cause]
ACTION ITEM: [Specific action]

[Continue until reaching systemic/process causes]
```

### Root Cause Analysis

#### What IS an Actionable Root Cause?
Answers that enable your team to:
1. **Prevent** reoccurrence of entire class of issue (improve MTBF)
2. **Reduce impact** if preventative measures fail (reduce MTTR)
3. **Enable investigation** if can't determine root cause (improve diagnosis)

#### What is NOT a Root Cause?
❌ Poor root causes:
- "CPU utilization became excessive"
  - Ask: Why? Why such impact? Why slow recovery?
- "Host ran out of memory"
  - Ask: Why? Why not caught in testing? Why not caught in prod? Why slow recovery?
- "Service X failed"
  - Ask: Why not more resilient? Why slow discovery/recovery?

✅ Key: **Take ownership**. Focus on effective action items for YOUR team.

### Action Items

#### Requirements
- **Name and description**: Clear what needs to be done
- **Owner**: Specific person/team
- **Related ticket**: TT or SIM assigned to owner
- **Completion date**: Realistic deadline
- **Priority**:
  - **High**: Root cause of outage OR hindered recovery
  - **Medium/Low**: Other improvements

#### Timeline Guidance
- **Most items**: Complete within 45 days
- **Long-term changes**: Create SIM backlog item instead
- **Overdue items**: Tracked as Policy Engine violations

#### Quality Standards
- Specific and achievable
- ✅ "COE tool API documentation available by May 2025"
- ❌ "We are going to create better communication"

### Lessons Learned
- 3-10 bulleted insights
- Focus on most valuable for others
- Link each lesson to action items
- Consider what other teams should learn
- Categories:
  - Technical (architecture, design, implementation)
  - Process (testing, deployment, change management)
  - Operational (monitoring, alerting, response)
  - Cultural (ownership, communication, priorities)

## Writing Guidelines

### General Rules
- Spell out acronyms on first use
- Be specific with times, dates, timezones
- Use format: "X of Y (Z percent)"
- Avoid customer names
- No confidential data
- No individual names (refer by role)
- Use COE tool (not Word docs or PDFs)

### Confidentiality
- **DON'T** mention customers by name
- **DON'T** speculate on customer business impact
- **DON'T** include legally privileged information
- **DO** use Secure COE feature when needed
- **DO** use Team-Only feature for SEV-2 events

### Title Guidelines
- State problem from customer perspective
- **NOT** the cause
- ✅ "Unable to receive multi-case"
- ❌ "Software change caused <problem>"

**Rationale**: 
1. Customer obsession (focus on customer problem)
2. Multiple causes at multiple levels (no single cause deserves title)

## Review Process

### Who Should Review?
1. **Team members** without direct context
2. **Principal engineers** and **senior managers** (based on impact)
3. **Director** (for high-impact incidents)
4. **External teams** (based on relevance)
5. **COE Bar Raiser** (escalation level)

### Review Period
- Minimum 24 hours (tool enforces)
- Account for worldwide public holidays
- Goal: Gather valuable feedback, not rush

### Responding to Feedback
- Capture email responses in COE body
- Address all issues raised
- Update COE based on feedback

## Common Pitfalls to Avoid

### Content Mistakes
❌ **DON'T**:
- Use vague terms ("significant," "some percentage," "few")
- Omit units or context
- Make readers do mental math
- Include customer names
- Speculate on customer business impact
- Let COEs languish incomplete
- Rush completion, compromising quality

✅ **DO**:
- Be specific with numbers and percentages
- Include context for all metrics
- State impact clearly and precisely
- Protect customer confidentiality
- Complete promptly with high quality

### Process Mistakes
❌ **DON'T**:
- Assign COEs punitively
- Refuse COEs asked for in good faith
- Skip COEs because "nothing to learn"
- Wait weeks to start
- Skip steps to meet deadline

✅ **DO**:
- Start immediately after impact mitigated
- Prioritize COE completion
- Complete within 14 days (typical)
- View as improvement opportunity
- Take ownership of your destiny

## Action Item Management

### Tracking Requirements
- Use COE tool's Action mechanism
- Track until completion
- Ensure acceptable progress
- **Overdue items** = Policy Engine violations

### Special Cases

**Already Complete?**
- Still create action item documenting:
  - What was done
  - Who did it
  - When completed
- Then resolve the action item
- **Purpose**: Tracking and organizational learning

**Software Deprecation?**
- **NO** to abandoning action items
- Take AI for replacement solution
- Ensure replacement avoids repeat incidents
- **Remember**: "It's not deprecated until it's deprecated"

**Orphaned Items?**
- Reassign from departed employees
- Items assigned to departed = orphaned = invisible in reports

## Amazon Leadership Principles via COE

1. **Earn Trust** (Vocally self-critical): Share findings and learnings
2. **Dive Deep**: Analyze all related processes and systems
3. **Insist on High Standards**: Enforce best practices via action items
4. **Ownership**: Make long-term improvements
5. **Deliver Results**: Complete action items timely
6. **Think Big**: Question assumptions, think long-term
7. **Customer Obsession**: Accurately analyze customer impact
8. **Invent and Simplify**: Improve/invent processes based on learnings
9. **Hire and Develop**: Develop employees to prevent recurrence

## Quick Decision Trees

### Should I Write a COE?
```
Did incident impact customers?
├─ YES → Write COE
└─ NO → Could it have impacted customers?
    ├─ YES → Consider writing COE
    └─ NO → Did it have valuable learnings?
        ├─ YES → Consider writing COE
        └─ NO → Probably skip, but if in doubt, write it
```

### Who Owns the COE?
```
Who owns addressing the root cause?
├─ Known → Assign to that manager
└─ Unknown → Assign to group responsible for investigation
    └─ Root cause found in different group → Reassign
```

### Multiple Teams Involved?
```
Should I contribute to another team's COE?
├─ Option 1: Contribute to their COE
│   └─ ⚠️ Lower quality outcomes
│   └─ ⚠️ More coordination overhead
└─ Option 2: Write your own COE ✅ RECOMMENDED
    └─ ✅ Better quality
    └─ ✅ Decentralized ownership
    └─ ✅ Each team focuses on their service
```

**It's OK to have multiple COEs for the same event!**

## Integration with Other Documents

### Related Internal Resources
- **[COE Documentation Guide](./coe_documentation_guide.md)** - Detailed structure and format
- **[Algorithm-Preserving Refactoring SOP](./algorithm_preserving_refactoring_sop.md)** - Specific to code refactoring
- **[Pytest Best Practices](./pytest_best_practices_and_troubleshooting_guide.md)** - Testing guidance

### When to Use Which Guide
- **This document**: Amazon-specific cultural practices and standards
- **COE Documentation Guide**: Detailed structure, templates, and examples
- **Refactoring SOP**: Technical refactoring with algorithm preservation

## Key Takeaways

1. **Blameless Culture**: COEs improve systems, not punish people
2. **Take Ownership**: Focus on what YOUR team can improve
3. **Be Specific**: Numbers, not vague terms
4. **Dive Deep**: Keep asking "why" until actionable
5. **Complete Promptly**: 14 days, information fresh
6. **Track Actions**: Use tool, ensure completion
7. **Share Learnings**: 2000+ Amazonians read COEs
8. **Customer First**: Analyze impact accurately, protect confidentiality
9. **Multiple COEs OK**: Each team owns their analysis
10. **If in Doubt, Write**: COEs are improvement opportunities

## FAQ Quick Reference

**Q: All SEV-1 events get COEs?**
A: Yes, with very few exceptions.

**Q: What about SEV-2?**
A: If customer-impacting or has valuable learnings, yes.

**Q: How long to complete?**
A: Target 14 days; don't let COEs languish.

**Q: Must have action items?**
A: Absolutely. Otherwise, why have Correction of Errors?

**Q: Items already done?**
A: Still create action item documenting what/who/when, then resolve.

**Q: Multiple teams involved?**
A: Each team should write their own COE (it's OK!).

**Q: Can I abandon AI if software deprecated?**
A: No. Take AI for replacement ensuring it prevents repeat incidents.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-19 | Initial document from Amazon COE Best Practices wiki | ML Platform Team |

---

## Additional Resources

### Related Documentation
- **[COE Documentation Guide](./coe_documentation_guide.md)** - Detailed COE structure and templates
- **[Algorithm-Preserving Refactoring SOP](./algorithm_preserving_refactoring_sop.md)** - Technical refactoring guidance
- **[Pytest Best Practices](./pytest_best_practices_and_troubleshooting_guide.md)** - Testing best practices

### External Resources
- Norm Kerth, "Project Retrospectives: A Handbook for Team Review" - Retrospective Prime Directive
- Five Whys Method: Wikipedia article on root cause analysis technique
- Site Reliability Engineering (Google): Chapter on postmortem culture

---

**Maintainer**: ML Platform Team  
**Last Review**: 2025-12-19  
**Next Review**: 2026-03-19

---

*This document summarizes best practices for Correction of Error analysis based on industry-leading incident management methodologies. For detailed structure and templates, see the [COE Documentation Guide](./coe_documentation_guide.md). For algorithm-specific refactoring guidance, see the [Algorithm-Preserving Refactoring SOP](./algorithm_preserving_refactoring_sop.md).*
