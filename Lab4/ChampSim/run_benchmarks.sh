./config.sh ece552/$1.json
make -j

bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/charlie.champsim.gz   >  output/$1.charlie &
bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/k_clique.champsim.gz  > output/$1.k_clique &
bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/llama2.champsim.gz    >   output/$1.llama2 &
bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/mcf.champsim.xz       >      output/$1.mcf &
bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/merced.champsim.gz    >   output/$1.merced &
bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/roms.champsim.xz      >     output/$1.roms &
bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/sssp.champsim.gz      >     output/$1.sssp &
bin/$1 --warmup-instructions 5000000 --simulation-instructions 20000000 /cad2/ece552f/Champsim_benchmarks/whisper.champsim.gz   >  output/$1.whisper

DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/1562901718646267924/BKBfaBBWm20yvuXF0IYLwfxCc-hDjJudWdxYmyabZxvS5F6yw6m6YTkVDZ7ykshvbLBK"

wait

python3 data_analysis.py --discord --webhook-url $DISCORD_WEBHOOK_URL $1
python3 global_comparison.py 
