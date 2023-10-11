#!/bin/bash -l

domains=(
	book
	arxiv
	c4
	cc/2019-30-head-en
	cc/2019-30-middle-en
	cc/2020-05-head-en
	cc/2020-05-middle-en
	cc/2021-04-head-en
	cc/2021-04-middle-en
	cc/2022-05-head-en
	cc/2022-05-middle-en
	cc/2023-06-head-en
	cc/2023-06-middle-en
	github
	)

# domains=(
# 	cc/2019-30-head-en
# 	cc/2019-30-middle-en
# 	cc/2020-05-head-en
# 	cc/2020-05-middle-en
# 	cc/2021-04-head-en
# 	cc/2021-04-middle-en
# 	cc/2022-05-head-en
# 	cc/2022-05-middle-en
# 	cc/2023-06-head-en
# 	cc/2023-06-middle-en 
# 	c4
# 	book
# )

# domains=(
# 	stack_exchange
# 	wiki
# 		)

block_size=${BLOCK:-6144}
train=${TRAIN:-1B}
validation=${VAL:-7M}

for d in ${domains[@]}; do
	echo domain $d block $block_size train $train validation $validation
	# BLOCK=$block_size TRAIN=$train VAL=$validation DOM=$d bash -l tokenize_pajama.sh
	BLOCK=$block_size TRAIN=$train VAL=$validation DOM=$d sbatch -J ${d}_${block_size} redpj_process/tokenize_pajama.sh
done

