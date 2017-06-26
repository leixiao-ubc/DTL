function [] = visualizeEstimation(nvis)

global Training result_dir Data

dnv = Training.data_normalization_value;

hFigure = figure(1); close(hFigure);
hFigure = figure(1);  set(hFigure,'visible','off');
for im = 1:nvis
  DataOne = Data{im};
  subaxis(3, nvis, im, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
  imagesc(max(0, min(1, customCrop(DataOne.Meas./dnv, Training.wpad))), [0,1]),colormap(gray), axis image, axis off;
  subaxis(3, nvis, im+nvis, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
  imagesc(max(0, min(1,customCrop(DataOne.ESTimg./dnv, Training.wpad))), [0,1]),colormap(gray), axis image, axis off;
  subaxis(3, nvis, im+nvis*2, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
  imagesc(max(0, min(1,DataOne.GTimg./dnv)), [0,1]),colormap(gray), axis image, axis off;
end

if(Training.isPreTraining)
    saveas(hFigure, sprintf('%s/ESTimg_pretrain_iter_%d_layer_%d', result_dir, Training.iter_training, Training.PreTrain.layer), 'png');
else
    saveas(hFigure, sprintf('%s/ESTimg_iter_%d', result_dir, Training.iter_training), 'png');
end
