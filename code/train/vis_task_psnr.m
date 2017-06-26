function [] = vis_task_psnr(psnr_all)

global Training

psnr_separate_noiselevel = reshape(psnr_all(:), Training.N./Training.num_task, Training.num_task);
psnr_separate_noiselevel = mean(psnr_separate_noiselevel, 1);
fprintf('mean psnr every %d images: ', Training.N./Training.num_task);
for idx_p = 1:Training.num_task
    fprintf('%.2f, ', psnr_separate_noiselevel(idx_p));
end
fprintf('\n');
    
end