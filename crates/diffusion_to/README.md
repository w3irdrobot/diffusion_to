# diffusion_to

Rust crate for interacting with the diffusion.to API.

## Using the crate

First, add the crate to the project.

```shell
cargo add diffusion_to
```

Instantiate a client, create an image request, and send it off. Wait for the image to be created and download it.

```rust
let client = DiffusionClient::new(args.api_key)?;

let request = ImageRequest::new(args.prompt)
    .update_steps(args.steps.try_into()?)
    .update_model(args.model.try_into()?)
let token = client.request_image(request).await?;

// wait for up to five minutes
let image = client
    .check_and_wait(token, Some(Duration::from_secs(300)))
    .await?;

println!("{}", image.raw)
```
