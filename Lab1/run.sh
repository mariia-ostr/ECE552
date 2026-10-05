cd lduh
make
cd ..

echo "Built PIN library"

gcc -S -O0 mbq1.c -o mbq1.S
cat mbq1.S

gcc -O0 mbq1.c -o mbq1

echo "Built Microbenchmark"

objdump --disassemble=main mbq1 > disassembly.txt
cat disassembly.txt

/cad2/ece552f/pin3.22 -t lduh/obj-intel64/lduh.so -- mbq1  80000000
/cad2/ece552f/pin3.22 -t lduh/obj-intel64/lduh.so -- mbq1 160000000
