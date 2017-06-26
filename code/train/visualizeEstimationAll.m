function [] = visualizeEstimationAll(nvis)

global Training result_dir Data

for TT = 1:nvis:Training.N
    hFigure = figure(1); close(hFigure);
         hFigure = figure(1);  set(hFigure,'visible','off');
         for im = 1:nvis
              blurry_idx = min(TT-1+im, Training.N);
              DataOne = Data{blurry_idx};
              subaxis(3, nvis, im, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
              imagesc(max(0, min(1, customCrop(DataOne.Meas./Training.data_normalization_value, Training.wpad))), [0,1]), axis image,colormap(gray), axis off;
              subaxis(3, nvis, im+nvis, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
              imagesc(max(0, min(1,customCrop(DataOne.ESTimg./Training.data_normalization_value, Training.wpad))), [0,1]), axis image,colormap(gray), axis off;  
              subaxis(3, nvis, im+nvis*2, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
              imagesc(max(0, min(1,DataOne.GTimg./Training.data_normalization_value)), [0,1]), axis image,colormap(gray), axis off;  
         end
         saveas(hFigure, sprintf('%s/ESTimg_stage%d_image_%02d_batch.png', result_dir, Training.iter_training, TT), 'png');
end