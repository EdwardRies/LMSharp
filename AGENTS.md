# LMSharp contributor guidance

Read the [README](README.md) and [architecture walkthrough](docs/Architecture.md) before changing behavior. Check the worktree and any applicable nested instructions, preserve unrelated edits, and use a focused task branch for changes. Keep the inference path readable to C# developers and use reference executables or Python only for development and verification.

- Use C# and .NET 10 with nullable reference types and implicit usings. Keep command-line behavior in the console project and numerical work behind explicit model or tensor boundaries.
- Preserve CPU operation without the CUDA runtime. Treat optional GPU and large-model checks as integration tests with explicit setup and skip reasons.
- Use async APIs for I/O, propagate cancellation where supported, and dispose resources with explicit ownership.
- Update focused tests for behavior changes and run the full suite. Build with `dotnet build LMSharp.slnx -c Release` and test with `dotnet test LMSharp.slnx -c Release --no-build`. Run asset-dependent checks when the affected behavior needs them. See [integration test setup](docs/Model-Compatibility.md#running-the-integration-tests).
- Review the final diff and report warnings, failures, and gaps in verification. Keep model weights, tokenizer downloads, local launch settings, and generated artifacts out of Git.
- Merge through a pull request after checks pass. Preserve shared history and avoid direct pushes to the default branch after the initial repository setup.
