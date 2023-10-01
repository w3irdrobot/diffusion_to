# diffusion.to CLI

Rust CLI for interacting with the diffusion.to API.

## Install

The CLI can easily be installed using `cargo`.

```shell
cargo install diffusion_to_cli
```

## Using the CLI

```shell
$ diffusion_to_cli --help
Rust CLI for interacting with the diffusion.to API

Usage: diffusion_to_cli [OPTIONS] --api-key <API_KEY> --prompt <PROMPT>

Options:
  -a, --api-key <API_KEY>          The token for the API
  -p, --prompt <PROMPT>            The prompt for the image
  -n, --negative <NEGATIVE>        The negative prompt for the image
  -s, --steps <STEPS>              The number of steps for the generation to use [default: 50] [possible values: 50, 100, 150, 200]
  -m, --model <MODEL>              The image model to use [default: beauty-realism] [possible values: beauty-realism, aesthetic-realism, anime-realism, analog-realism, dream-reality, stable-diffusion, toon-animated, fantasy-animated]
      --size <SIZE>                The size of the image [default: small] [possible values: small, medium, large]
  -o, --orientation <ORIENTATION>  The orientation of the image [default: square] [possible values: square, landscape, portrait]
      --out <OUT>                  The file to output the image to
  -h, --help                       Print help
  -V, --version                    Print version
```
