function [] = saveInput()

global Training result_dir Data

dnv = Training.data_normalization_value;
dir = sprintf('%s/training_input/', result_dir);
mkdir(dir);
for blurry_idx = 1:Training.N
    imwrite(Data{blurry_idx}.Meas./dnv, sprintf('%s/input_%02d.%s', dir, blurry_idx, 'png'));
end 

end