FROM node:24-slim

# Workdir where Dev Containers will mount this git repo in the container
WORKDIR /workspaces/MinSampleTemplates/WebpackHtmlWithJavaScript

# Unlike the Python samples, dependencies are not installed in the image:
# node_modules must live in the project folder, which is mounted over WORKDIR.
# devcontainer.json runs `npm ci` after the container is created instead.

CMD ["bash"]
