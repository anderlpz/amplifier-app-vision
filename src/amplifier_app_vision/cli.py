"""CLI interface for vision analysis."""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_time=False)],
    )


@click.command()
@click.argument("sources", nargs=-1, required=False)
@click.option("--prompt", "-p", type=str, default="What's in this image?", help="Analysis prompt")
@click.option("--describe", "-d", is_flag=True, help="Get detailed description")
@click.option("--extract-text", "-t", is_flag=True, help="Extract text from image (OCR)")
@click.option("--quality", "-q", type=click.Choice(["quick", "normal", "detailed", "full"]), default="normal", help="Image quality preset")
@click.option("--model", "-m", type=str, default="gpt-4o", help="OpenAI model to use")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Save analysis to file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(
    sources: tuple[str] | None,
    prompt: str,
    describe: bool,
    extract_text: bool,
    quality: str,
    model: str,
    output: Path | None,
    verbose: bool,
):
    """Analyze images using OpenAI GPT-4 Vision.

    SOURCES can be file paths or URLs to images.

    Examples:

        vision image.png  # Analyze single image

        vision image1.png image2.png  # Compare multiple images

        vision screenshot.png -p "What code is shown?"

        vision document.png --extract-text  # OCR

        vision photo.jpg --describe  # Detailed description
    """
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging(verbose)
    
    # Require sources for CLI analysis
    if not sources:
        console.print("[red]Error: Provide at least one image path or URL[/red]")
        console.print("\nUsage: vision IMAGE [IMAGE...] [OPTIONS]")
        sys.exit(1)
    
    # Validate API key
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
        console.print("\nSet your API key:")
        console.print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    # Show banner
    console.print(
        Panel.fit(
            "[bold cyan]Vision[/bold cyan]\n"
            "Image analysis powered by OpenAI GPT-4 Vision",
            border_style="cyan",
        )
    )
    
    try:
        from .analyzer import VisionAnalyzer
        
        analyzer = VisionAnalyzer(model=model, quality=quality)
        
        # Configuration summary
        console.print(f"\n[cyan]Configuration:[/cyan]")
        console.print(f"  Model: {model}")
        console.print(f"  Quality: {quality}")
        console.print(f"  Images: {len(sources)}")
        
        results = []
        
        if len(sources) == 1:
            # Single image analysis
            source = sources[0]
            console.print(f"\n[cyan]Analyzing:[/cyan] {source}")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Processing...", total=None)
                
                if describe:
                    result = {"text": analyzer.describe(source), "source": source}
                elif extract_text:
                    result = {"text": analyzer.extract_text(source), "source": source}
                else:
                    result = analyzer.analyze(source, prompt=prompt)
                    result["source"] = source
            
            results.append(result)
            
        else:
            # Multiple images - analyze together
            console.print(f"\n[cyan]Analyzing {len(sources)} images together...[/cyan]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Processing...", total=None)
                result = analyzer.analyze_multiple(list(sources), prompt=prompt)
            
            results.append(result)
        
        # Display results
        for result in results:
            _display_result(result, console)
        
        # Save if output specified
        if output and results:
            _save_results(results, output)
            console.print(f"\n[green]Saved to {output}[/green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _display_result(result: dict, console: Console):
    """Display analysis result."""
    text = result.get("text", "")
    
    # Show source if available
    source = result.get("source", "")
    title = f"Analysis: {source}" if source else "Analysis"
    
    console.print(Panel(text, title=title, border_style="green"))
    
    # Show usage stats if available
    usage = result.get("usage")
    if usage:
        console.print(f"[dim]Tokens: {usage.get('total_tokens', 0):,} "
                     f"(prompt: {usage.get('prompt_tokens', 0):,}, "
                     f"completion: {usage.get('completion_tokens', 0):,})[/dim]")
    
    # Show image metadata
    metadata = result.get("image_metadata") or result.get("images_metadata")
    if metadata:
        if isinstance(metadata, list):
            for i, m in enumerate(metadata, 1):
                if m.get("filename"):
                    console.print(f"[dim]Image {i}: {m.get('filename')} "
                                 f"({m.get('optimized_size', 0):,} bytes)[/dim]")
        elif metadata.get("filename"):
            console.print(f"[dim]Image: {metadata.get('filename')} "
                         f"({metadata.get('optimized_size', 0):,} bytes, "
                         f"~{metadata.get('estimated_tokens', 0):,} tokens)[/dim]")


def _save_results(results: list, output: Path):
    """Save results to file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        for i, result in enumerate(results, 1):
            if len(results) > 1:
                f.write(f"=== Result {i} ===\n\n")
            
            source = result.get("source", "")
            if source:
                f.write(f"Source: {source}\n\n")
            
            f.write(result.get("text", ""))
            f.write("\n\n")
            
            usage = result.get("usage")
            if usage:
                f.write(f"Tokens: {usage.get('total_tokens', 0)}\n")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
