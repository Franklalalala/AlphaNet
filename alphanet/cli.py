import click
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from rich.panel import Panel
from .train import run_training
from pyfiglet import Figlet
from alphanet.config import All_Config

console = Console()

def display_header():
    
    f = Figlet(font='slant', width=120)  
    ascii_art = Text(f.renderText('ALPHANET'), style="bold bright_magenta")
    gradient_text = Text.assemble(
    ("██╗  ██╗", "gradient(90, #FF00FF, #00FFFF)"),  
    (" ██████╗ ", "bold white on #2E0854"),          
    ("\n" + "═"*50, "bright_cyan")                   
    )  
    console.print(
     Panel.fit(
        Text.assemble(ascii_art, "\n\n", gradient_text),
        title="[blink]🚀 AlphaNet Training 🚀[/]",  
        subtitle="[italic bright_white]Deep Learning Force Field Toolkit[/]",  
        box=box.ROUNDED,
        border_style="bright_blue",
        padding=(1, 4),
        width=100
    )       )

def display_config_table(main_config, runtime_config):
   
    table = Table(title="Configuration Overview", show_header=True, header_style="bold cyan")
    table.add_column("Section", style="dim", width=20)
    table.add_column("Key", style="dim")
    table.add_column("Value", justify="right")
    
    for section, values in main_config.items():
        table.add_row("[bold]Main Config[/]", f"[bold]{section}[/]", "")
        for k, v in values.items():
            table.add_row("", k, str(v))
    
    table.add_row("[bold]Runtime Config[/]", "num_nodes", str(runtime_config["num_nodes"]))
    table.add_row("", "num_devices", str(runtime_config["num_devices"]))
    table.add_row("", "resume", str(runtime_config["resume"]))
    table.add_row("", "ckpt_path", str(runtime_config["ckpt_path"]))
    
    console.print(table)

@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--num_nodes", type=int, default=1, help="Number of machines (nodes)")
@click.option("--num_devices", type=int, default=1, help="GPUs per node")
@click.option("--resume", is_flag=True, help="Resume training from checkpoint")
@click.option("--ckpt_path", type=click.Path(), default=None, help="Path to checkpoint file")
def main(config, num_nodes, num_devices, resume, ckpt_path):
   
    with open(config, "r") as f:
        mconfig = json.load(f)
    main_config = All_Config().from_json(config)
    
    runtime_config = {
        "num_nodes": num_nodes,
        "num_devices": num_devices,
        "resume": resume,
        "ckpt_path": ckpt_path
    }
    
    display_header()
    display_config_table(mconfig, runtime_config)
    
    #merged_config = {**main_config, **runtime_config}
    run_training(main_config, runtime_config)

if __name__ == "__main__":
    main()

