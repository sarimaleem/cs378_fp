# bash script that gets all the ambiguous examples

python3 run_cartography_models.py --do_eval --task nli --dataset snli --model ./trained_model/checkpoint-500 --output_dir ./model500/ --max_eval_samples=50000
python3 run_cartography_models.py --do_eval --task nli --dataset snli --model ./trained_model/checkpoint-1000 --output_dir ./model1000/ --max_eval_samples=50000
python3 run_cartography_models.py --do_eval --task nli --dataset snli --model ./trained_model/checkpoint-1500 --output_dir ./model1500/ --max_eval_samples=50000
python3 run_cartography_models.py --do_eval --task nli --dataset snli --model ./trained_model/checkpoint-2000 --output_dir ./model2000/ --max_eval_samples=50000
python3 run_cartography_models.py --do_eval --task nli --dataset snli --model ./trained_model/checkpoint-2500 --output_dir ./model2500/ --max_eval_samples=50000
python3 run_cartography_models.py --do_eval --task nli --dataset snli --model ./trained_model/checkpoint-3000 --output_dir ./model3000/ --max_eval_samples=50000
python3 run_cartography_models.py --do_eval --task nli --dataset snli --model ./trained_model/ --output_dir ./model/ --max_eval_samples=50000