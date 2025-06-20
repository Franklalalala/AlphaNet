import torch
import click
from pathlib import Path
from typing import Optional

@click.command()
@click.option("--input", "-i", required=True, type=click.Path(exists=True),
              help="Path to input checkpoint file")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Path to save modified checkpoint")
@click.option("--prefix", "-p", default="l.model",
              help="Prefix to replace in state_dict keys")
@click.option("--replacement", "-r", default="l", show_default=True,
              help="Replacement string for the prefix")
@click.option("--force", "-f", is_flag=True,
              help="Overwrite output file if exists")
@click.option("--verbose", "-v", is_flag=True,
              help="Show detailed operation logs")
def cli(input: str, output: str, prefix: str, replacement: str,
        force: bool, verbose: bool) -> None:
    """
    Modify PyTorch checkpoint key prefixes with CLI interface
    
    Examples:
    
    \b
    # Basic usage
    ckpt-modifier -i input.ckpt -o output.ckpt -p "model." -r "new_model."
    
    \b
    # Remove prefix
    ckpt-modifier -i model.ckpt -o modified.ckpt -p "module." -r ""
    """
    # Validate output path
    output_path = Path(output)
    if output_path.exists() and not force:
        raise click.ClickException(f"Output file {output} already exists. Use --force to overwrite.")

    # Execute core operation
    try:
        modified = process_checkpoint(
            input_path=input,
            output_path=output,
            prefix_to_replace=prefix,
            replacement=replacement,
            verbose=verbose
        )
    except Exception as e:
        raise click.ClickException(f"Operation failed: {str(e)}")

    # Print summary
    click.secho("\nOperation Summary:", fg='green', bold=True)
    click.echo(f"• Input checkpoint:   {input}")
    click.echo(f"• Output checkpoint:  {output}")
    click.echo(f"• Modified keys:       {modified}")
    click.echo(f"• Prefix replacement: '{prefix}' → '{replacement}'")

def process_checkpoint(input_path: str, output_path: str,
                       prefix_to_replace: str, replacement: str,
                       verbose: bool = False) -> int:
    """Core processing logic with error handling"""
    # Load checkpoint
    if verbose:
        click.secho(f"\nLoading checkpoint from {input_path}...", fg='yellow')
    
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

    # Verify state_dict structure
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint missing required 'state_dict' key")

    original_keys = checkpoint['state_dict'].keys()
    if verbose:
        click.echo(f"Found {len(original_keys)} keys in state_dict")

    # Perform key replacement
    new_state_dict = {}
    modified_count = 0
    
    for k, v in checkpoint['state_dict'].items():
        new_key = k.replace(prefix_to_replace, replacement)
        if new_key != k:
            modified_count += 1
            if verbose:
                click.echo(f"  {k} → {new_key}")
        new_state_dict[new_key] = v

    # Validate modifications
    if modified_count == 0:
        raise ValueError(f"No keys contained prefix '{prefix_to_replace}'")

    # Save updated checkpoint
    try:
        if verbose:
            click.secho(f"\nSaving modified checkpoint to {output_path}...", fg='yellow')
        
        torch.save(new_state_dict, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint: {str(e)}")

    return modified_count

if __name__ == "__main__":
    cli()

