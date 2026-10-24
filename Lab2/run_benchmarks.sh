cd ChampSim

./config.sh ece552/$1.json
make

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/charlie.champsim.gz | grep Mispredict > ../charlie.$1.mispred

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/mcf.champsim.xz | grep Mispredict > ../mcf.$1.mispred

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/roms.champsim.xz | grep Mispredict > ../roms.$1.mispred

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/merced.champsim.gz | grep Mispredict > ../merced.$1.mispred

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/k_clique.champsim.gz | grep Mispredict > ../k_clique.$1.mispred

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/sssp.champsim.gz | grep Mispredict > ../sssp.$1.mispred

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/llama2.champsim.gz | grep Mispredict > ../llama2.$1.mispred

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/whisper.champsim.gz | grep Mispredict > ../whisper.$1.mispred


