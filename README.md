# diffusion_to

Rust crate and CLI for interacting with the diffusion.to API.

## Using the crate

First, add the crate to the project.

```shell
cargo add diffusion_to
```

Instantiate a client, create an image request, and send it off. Wait for the image to be created and download it.

```rust
let client = DiffusionClient::new(args.api_key)?;

let mut request = ImageRequest::new(args.prompt)
    .update_steps(args.steps.try_into()?)
    .update_model(args.model.try_into()?)
let token = client.request_image(request).await?;

// wait for up to five minutes
let image = client
    .check_and_wait(token, Some(Duration::from_secs(300)))
    .await?;

println!("{}", iamge.raw)
```

## Using the CLI

```shell
cargo install diffusion_to
```

```
CLI for requesting and downloading AI-created images via diffusion.to

Usage: diffusion_to [OPTIONS] --api-key <API_KEY> --prompt <PROMPT>

Options:
  -a, --api-key <API_KEY>          The token for the API
  -p, --prompt <PROMPT>            The prompt for the image
  -n, --negative <NEGATIVE>        The negative prompt for the image
  -s, --steps <STEPS>              The number of steps for the generation to use [default: 50]
  -m, --model <MODEL>              The image model to use [default: beauty_realism]
      --size <SIZE>                The size of the image [default: small]
  -o, --orientation <ORIENTATION>  The orientation of the image [default: square]
  -h, --help                       Print help
  -V, --version                    Print version
```
