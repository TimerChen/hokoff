export GAMECORE_PATH=`pwd`
export WINEPATH="${GAMECORE_PATH}/lib/;${GAMECORE_PATH}/bin/"

./gamecore-server-linux-amd64 server --server-address :23333 \
    --simulator-remote-bin bin/sgame_simulator_remote_zmq \
    --simulator-repeat-bin bin/sgame_simulator_repeated_zmq
