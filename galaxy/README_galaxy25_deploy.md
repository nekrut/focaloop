# Dedicated Galaxy 25.0+ server for focaloop

This directory holds a ready-to-run configuration for a Galaxy `25.0+` instance that serves the
focaloop tools both ways: as ToolShed-style XML tools in a **Focaloop** tool panel section (`tools/focaloop/*.xml`), and as
unprivileged user-defined tools (UDTs) that users register from `blueprints/*/galaxy_udt.yml`.

| File | Purpose |
|------|---------|
| `galaxy_25_focaloop.yml` | `galaxy.yml` template: gravity on 127.0.0.1, closed registration, SQLite, UDTs on, Apptainer-only jobs |
| `focaloop_tool_conf.xml` | Tool panel section pointing at `tools/focaloop/{expressionloop,variationloop,geneloop,focaloop_query}.xml` in this checkout |
| `setup_galaxy25.sh` | Clones Galaxy `release_25.0`, renders the config, pulls the focaloop image, probes HTTP 206 |

## Targets

| Host | `GALAXY_ROOT` | Notes |
|------|---------------|-------|
| garage | `/media/anton/hd2/galaxy-focaloop` | default; data, jobs and the `.sif` live on `hd2` |
| cluster | `$HOME/galaxy-focaloop` (or scratch) | same script; set `GALAXY_ROOT` |

## Steps

1. Build and push the image (from the repository root), or build the `.sif` locally:
   ```bash
   docker build -f container/Dockerfile -t ghcr.io/galaxyproject/focaloop:0.3.0 .
   docker push ghcr.io/galaxyproject/focaloop:0.3.0          # needs push rights to the org
   ```
   Without push rights, build the `.sif` on the host and skip the pull:
   `apptainer build $GALAXY_ROOT/images/focaloop_0.3.0.sif docker-daemon://ghcr.io/galaxyproject/focaloop:0.3.0`.
2. Install: `GALAXY_ROOT=/media/anton/hd2/galaxy-focaloop galaxy/setup_galaxy25.sh`.
3. Start: `GALAXY_CONFIG_FILE=$GALAXY_ROOT/config/galaxy.yml $GALAXY_ROOT/server/.venv/bin/galaxyctl start`.
   Gunicorn binds `127.0.0.1:8080` only; reach it through an nginx TLS reverse proxy or
   `ssh -L 8080:127.0.0.1:8080 garage`.
4. Registration is closed (`allow_user_creation: false`). Create the admin `anton@nekrut.org` by starting
   once with `GALAXY_CONFIG_OVERRIDE_ALLOW_USER_CREATION=true`, registering, and restarting without it;
   later accounts are created under **Admin → Users**. Grant UDT creators the role of type
   `user_tool_execute` (`Role.types.USER_TOOL_EXECUTE`); see
   <https://docs.galaxyproject.org/en/master/admin/user_defined_tools.html>.
5. Register the UDTs as a user (any of):
   - Galaxy MCP: `create_user_tool` with the parsed `blueprints/<id>/galaxy_udt.yml`, then `run_user_tool`;
   - REST: `POST /api/unprivileged_tools` with the parsed YAML as the tool representation (see the UDT docs for the payload).

## Design choices

- **Containers only.** `require_container: true` and `conda_auto_install: false`: every job runs in
  `ghcr.io/galaxyproject/focaloop:0.1.0` via Apptainer, which is also what a UDT demands.
- **Outbound Range access.** Apptainer shares the host network namespace, so `focaloop` can issue
  HTTPS `Range` requests to ENA (tier 3), `sra-pub-run-odp` on S3 (tier 2) and remote CRAM/BAM hosts
  (tier 1). `setup_galaxy25.sh` probes ENA and prints the HTTP status (expect `206`).
- **No TPV.** All tools, including `tool_type_user_defined`, land in `apptainer_local`. If TPV is added
  later, accept `tool_type_user_defined` on the destination (gotcha 3 below).
- **Builtin blueprints.** The image sets `FOCALOOP_BLUEPRINT_PATH=/opt/focaloop/blueprints`, so UDTs call
  `focaloop run --blueprint panexpr`; XML tools pass `$__tool_directory__/blueprint.focaloop.yml`.

## Operational gotchas (verified on Galaxy 27.0 / Slurm + Apptainer)

1. **No bash `$(cmd)` in `shell_command`.** `cwl_utils` parses every `$(...)` in a `GalaxyUserTool`
   `shell_command` as a CWL JavaScript expression, so `$(pwd)` or `$(date)` fails with a `ReferenceError`.
   Use `$PWD`, plain shell variables, or backticks. (Arithmetic `$(( ))` belongs only in XML/Cheetah
   wrappers.) The four `galaxy_udt.yml` files here use only `$(inputs.…)` expressions.
2. **No `docker://` prefix on `container:`.** `explicit_singularity` prepends `docker://` itself; writing
   `container: docker://ghcr.io/…` yields `docker://docker://ghcr.io/…` and the pull fails. Write the bare
   reference: `container: ghcr.io/galaxyproject/focaloop:0.1.0`.
3. **TPV rejects UDTs by default.** In `tpv/core/mapper.py` every `Destination` injects
   `reject: [tool_type_user_defined]`. A Slurm/Apptainer destination that should run UDTs must list it
   explicitly:
   ```yaml
   destinations:
     slurm_apptainer:
       scheduling:
         accept: [tool_type_user_defined]
   ```
   The submitting user also needs a role of type `user_tool_execute` (`Role.types.USER_TOOL_EXECUTE`);
   without it creation/run is refused regardless of routing.

## Not done here

Nothing in this directory has been executed against a live server: the image is not yet published,
and bringing up a server on garage or cluster waits for sign-off.
