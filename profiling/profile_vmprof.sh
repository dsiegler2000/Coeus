now=`date +"%m-%d-%Y %H:%M:%S"`
filename="profiling/profiling_vmprof_${now}.log"
pypy -m vmprof -o "${filename}" src/communication.py --profile "r5k1/1p4pp/1p6/2pp2P1/4rb2/P1P2N2/1P1B3P/4RK2 w - - 0 27"
echo "Profile saved to ${filename}, use vmprofshow '${filename}' to view it"