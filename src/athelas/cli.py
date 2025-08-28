"""Command-line interface for Athelas."""
import click
from .__version__ import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """Athelas: Zettelkasten-inspired ML model catalog."""
    pass


@main.command()
@click.argument('query')
def search(query):
    """Search for components in the knowledge base."""
    click.echo(f"Searching for: {query}")
    click.echo("Note: Knowledge retriever not yet implemented.")
    # Future implementation:
    # from .knowledge.retriever import KnowledgeRetriever
    # retriever = KnowledgeRetriever()
    # results = retriever.search(query)
    # for result in results:
    #     click.echo(f"- {result['component_id']}: {result['text'][:100]}...")


@main.command()
def validate():
    """Validate the knowledge system."""
    click.echo("Validating knowledge system...")
    click.echo("Note: Knowledge orchestrator not yet implemented.")
    # Future implementation:
    # from .knowledge.orchestrator import KnowledgeOrchestrator
    # orchestrator = KnowledgeOrchestrator()
    # orchestrator.validate_knowledge_system()
    # click.echo("Knowledge system validation complete.")


@main.command()
def info():
    """Show package information."""
    click.echo(f"Athelas version: {__version__}")
    click.echo("Zettelkasten-inspired ML model catalog with intelligent knowledge management")
    click.echo("Repository: https://github.com/TianpeiLuke/athelas")


@main.command()
def list_models():
    """List available models in the catalog."""
    click.echo("Available model categories:")
    click.echo("- Lightning models: PyTorch Lightning implementations")
    click.echo("- PyTorch models: Native PyTorch implementations")
    click.echo("- XGBoost models: Gradient boosting models")
    click.echo("- LightGBM models: Fast gradient boosting")
    click.echo("- Actor-critic models: Reinforcement learning")
    click.echo("- Bandit models: Multi-armed bandit algorithms")


@main.command()
def list_processors():
    """List available data processors."""
    click.echo("Available processor categories:")
    click.echo("- Text processors: BERT tokenization, Gensim, etc.")
    click.echo("- Tabular processors: Numerical imputation, categorical encoding")
    click.echo("- Image processors: Computer vision preprocessing")
    click.echo("- Feature processors: Feature engineering components")
    click.echo("- Augmentation processors: Data augmentation techniques")


if __name__ == '__main__':
    main()
