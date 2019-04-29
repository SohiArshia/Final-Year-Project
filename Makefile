clean:
	rm -f neat-checkpoint*
	rm -f *.obj
	rm -f *.csv
	rm -f *.svg
	rm -f *.gv
	rm -f gen_counter.txt

neat:
	bash master_neat.sh

hyper:
	bash master_hyper.sh

es_hyper:
	bash master_es_hyper.sh

die:
	killall -9 fceux