
bash ./imgs_to_groups_fake.sh

mkdir -p results

matlab -nodisplay -nodesktop -r "run Intra_niqe_test_utkface.m"
