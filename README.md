# One2Branch

## The code of branching decoder

We implement branching decoder based on the code of T5 (transformers\models\t5\modeling_t5.py).
The code of the architecture of One2Branch is  transformers\models\t5\modeling_t5_branching.py
The code of decoding One2Branch is the function 'branching_decoding()' in 'transformers\generation\utils.py'  


## How to run 
run_one2branch_kg.py is the code for one2branch training and inference.
You use the script 'script_branch_kp.sh' to run it 

```
bash script_branch_kp.sh
```