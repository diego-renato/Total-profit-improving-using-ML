campanias_path = 'D:\Diego archivos\Desktop\concursos\interbank\ib_base_campanias\ib_base_campanias.csv'
digital_path = 'D:\Diego archivos\Desktop\concursos\interbank\ib_base_campanias\ib_base_campanias.csv'


get_campanias:
	python ./code/campanias_treatment.py \
            --dataset $(campanias_path)

get_digital:
	python ./code/digital_treatment.py \
            --dataset $(digital_path)


