"""
Multi-hop research synthesizer module.

Three-stage pipeline:
  1. Extract: per-source key finding extraction (runs once per source)
  2. Cross-reference: identify consensus, contradictions, and gaps across sources
  3. Format: produce structured final output
"""

import dspy


# -- Stage signatures (hardcoded, not built from task_config.fields) ---------

class SourceExtraction(dspy.Signature):
    """Extract key findings from a single source document relevant to the research question."""
    guidelines: str = dspy.InputField(desc="Research synthesis guidelines")
    source_text: str = dspy.InputField(desc="A single source document")
    research_question: str = dspy.InputField(desc="The question to research")
    key_findings: str = dspy.OutputField(
        desc="Numbered list of key findings from this source, each on its own line"
    )


class CrossReference(dspy.Signature):
    """Analyze findings from multiple sources to identify consensus, contradictions, and gaps."""
    guidelines: str = dspy.InputField(desc="Research synthesis guidelines")
    all_findings: str = dspy.InputField(desc="Key findings from all sources, labeled by source number")
    research_question: str = dspy.InputField(desc="The question to research")
    consensus: str = dspy.OutputField(desc="Points that multiple sources agree on")
    contradictions: str = dspy.OutputField(
        desc="Pairs of conflicting claims with source attribution, e.g. 'Source 1 says X, but Source 3 says Y'"
    )
    gaps: str = dspy.OutputField(desc="Topics or questions that no source addresses")


class StructuredOutput(dspy.Signature):
    """Format the synthesis into a structured research report."""
    guidelines: str = dspy.InputField(desc="Research synthesis guidelines")
    consensus: str = dspy.InputField(desc="Points of agreement across sources")
    contradictions: str = dspy.InputField(desc="Conflicting claims between sources")
    gaps: str = dspy.InputField(desc="Gaps in the research coverage")
    research_question: str = dspy.InputField(desc="The original research question")
    key_findings: str = dspy.OutputField(
        desc="Numbered list of key findings with source attribution"
    )
    contradictions_report: str = dspy.OutputField(
        desc="Formatted pairs of conflicting claims with source numbers"
    )
    gaps_report: str = dspy.OutputField(
        desc="List of topics no source addresses"
    )
    summary: str = dspy.OutputField(
        desc="Executive summary paragraph (3-5 sentences)"
    )


SOURCE_SEPARATOR = "===SOURCE_SEPARATOR==="


# -- Module ------------------------------------------------------------------

class ResearchSynthesizerModule(dspy.Module):
    def __init__(self, task_config):
        super().__init__()
        self.guidelines = task_config.guidelines
        self.extract = dspy.ChainOfThought(SourceExtraction)
        self.cross_ref = dspy.ChainOfThought(CrossReference)
        self.format_output = dspy.ChainOfThought(StructuredOutput)

    def forward(self, sources: str, research_question: str):
        # Stage 1: per-source extraction
        source_blocks = [s.strip() for s in sources.split(SOURCE_SEPARATOR) if s.strip()]
        all_findings = []
        for i, src in enumerate(source_blocks):
            result = self.extract(
                guidelines=self.guidelines,
                source_text=src,
                research_question=research_question,
            )
            all_findings.append(f"[Source {i + 1}]:\n{result.key_findings}")

        combined_findings = "\n\n".join(all_findings)

        # Stage 2: cross-reference
        cross = self.cross_ref(
            guidelines=self.guidelines,
            all_findings=combined_findings,
            research_question=research_question,
        )

        # Stage 3: structured output
        final = self.format_output(
            guidelines=self.guidelines,
            consensus=cross.consensus,
            contradictions=cross.contradictions,
            gaps=cross.gaps,
            research_question=research_question,
        )

        return dspy.Prediction(
            key_findings=final.key_findings,
            contradictions=final.contradictions_report,
            gaps=final.gaps_report,
            summary=final.summary,
        )
