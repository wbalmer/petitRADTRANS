# Build the petitRADTRANS build docker image for the Gitlab repository, then push it
echo "Starting building petitRADTRANS docker build image..."
echo "Authenticating to registry.gitlab.com, use your Gitlab Username and Password"
docker login registry.gitlab.com

printf "\nBuilding image..."
docker build -t registry.gitlab.com/mauricemolli/petitradtrans:build .

printf "\nPushing image to gitlab..."
docker push registry.gitlab.com/mauricemolli/petitradtrans:build

echo "Done."
