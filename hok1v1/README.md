# Start Sample
```bash
sh offline_sample/scripts/start_sample.sh <levels_str> <eval_num> <cpu_num> <datasets_repo_name>`<backend>` 
bash offline_sample/scripts/start_sample1.sh 1,1 20 1 1v1version1 norm_medium
bash offline_sample/scripts/start_sample.sh 5,5 3 10 1v1version1 norm_medium
```

## sample time
```
# 40, 25
Current done actor num: 25, time: 12803
# 3, 10
Total actor num:10
Current done actor num: 10, time: 541
```

## Start Gamecore
```bash
export GAMECORE_PATH=`pwd`/rl_framework/gamecore/
export WINEPATH="${GAMECORE_PATH}/lib/;${GAMECORE_PATH}/bin/"
wine $GAMECORE_PATH/gamecore-server.exe server --server-address :23333
./gamecore-server-linux-amd64 server --server-address :23333
```
