import torch
import click
from pathlib import Path
from typing import Optional
from alphanet.models.model import AlphaNetWrapper
from alphanet.config import All_Config

@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to model configuration JSON file")
@click.option("--checkpoint", "-m", required=True, type=click.Path(exists=True),
              help="Path to trained model checkpoint")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output path for frozen model")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu",
              show_default=True, help="Device for model loading")
@click.option("--force", "-f", is_flag=True,
              help="Overwrite existing output file")
@click.option("--verbose", "-v", is_flag=True,
              help="Show detailed processing information")
def freeze_model(config: str, checkpoint: str, output: str, device: str,
                 force: bool, verbose: bool) -> None:
    """
    Freeze trained neural network models for production deployment
    
    Features:
    - Automatic device detection
    - Model architecture validation
    - TorchScript optimization
    - Cross-platform compatibility
    
    Examples:
    
    \b
    # Basic usage
    model-freezer -c config.json -m model.ckpt -o frozen_model.pt
    
    \b
    # Force overwrite and verbose mode
    model-freezer -c config.json -m model.ckpt -o /tmp/frozen.pt -fv
    """
    # Validate output path
    output_path = Path(output)
    if output_path.exists() and not force:
        raise click.ClickException(f"Output file {output} exists. Use --force to overwrite.")

    try:
        if verbose:
            click.secho("\n[1/4] Loading configuration...", fg='cyan')
            click.echo(f"Config path: {config}")
            
        config_obj = All_Config().from_json(config)
        
        if verbose:
            click.secho("[2/4] Initializing model...", fg='cyan')
            click.echo(f"Device: {device.upper()}\nModel architecture:")
            click.echo(config_obj.model)
            
        model = AlphaNetWrapper(config_obj).to(device)
        
        if verbose:
            click.secho("[3/4] Loading weights...", fg='cyan')
            click.echo(f"Checkpoint: {checkpoint}")
            
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        
        if verbose:
            click.secho("[4/4] Freezing model...", fg='cyan')
            
        #with torch.no_grad():
        script_model = torch.jit.script(model.eval())
        optimized_model = torch.jit.optimize_for_inference(script_model)
            
        optimized_model.save(output)
        
        click.secho(f"\n✅ Successfully saved frozen model to {output}", fg='green', bold=True)
        if verbose:
            click.echo(f"Model size: {output_path.stat().st_size/1e6:.1f} MB")
            
    except Exception as e:
        error_msg = f"\n❌ Freezing failed: {str(e)}"
        click.secho(error_msg, fg='red', bold=True)
        raise click.Abort()

if __name__ == "__main__":
    freeze_model()

