# Updating the gitlab CI

The Gitlab CI currently has two stages: the first builds the conda environment for mcvine.acc (`build-conda-env`) and the second runs the unit tests (`build-job-using-docker`). The CI file is at `.gitlab-ci.yml`.

Both Gitlab CI stages are run within a Docker container stored in the mcvine.acc Gitlab project container repo.

Since building the environment can take a long time, the first stage to build the conda environment is usually run once and cached on the Gitlab runner. If the cached environment exists, then the stage is skipped and moves on to the second stage.

## Building the Docker image

The Dockerfile for the CI is located at `.gitlab/docker/Dockerfile`. This image is setup to use the pre-built NVIDIA CUDA images which have Ubuntu + CUDA installed already (see https://hub.docker.com/r/nvidia/cuda). Some basic packages are installed via apt-get and mamba is downloaded and installed using Mambaforge.

The Docker image should rarely have to be updated, usually only to use newer versions of the CUDA toolkit and/or Ubuntu.

To build the Docker image and tag it for the Gitlab project:

```
cd .gitlab/docker/
docker build -t code.ornl.gov:4567/mcvine/acc .
```

## Pushing the image to Gitlab

After the image has been built, it needs to be pushed to the container repo in Gitlab so the runner can use it in the CI.

Login to Gitlab with

```
docker login code.ornl.gov:4567
```

After authenticating, the Docker image can be uploaded with
```
docker push code.ornl.gov:4567/mcvine/acc
```

## Updating the Gitlab runner cached environment

Anytime the Docker image changes and/or the conda environment changes, the Gitlab runner cache needs to be cleaned so that the environment can be re-created.

In Gitlab, navigate to the "CI/CD" -> "Pipelines" tab. On the top left, click the "Clear runner caches" button. If the button does not appear, then you need to be sure you have the correct privileges in the Gitlab project.

Once the runner cache has been cleared, clicking the "Run pipeline" button in the top right will force the first stage to create the conda environment.

