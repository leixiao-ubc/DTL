function [] = visualizeFilters()

global Training Model result_dir 

vis_grid_h = Model.filterWidth-1;
vis_grid_w = Model.filterWidth+1;

for layer = 1:Model.numDenoiseLayers    
    hFigure = figure(1); close(hFigure);
    hFigure = figure(1);  set(hFigure,'visible','off');
    for yy = 1:vis_grid_h
         for xx = 1:vis_grid_w
             i = (yy-1)*vis_grid_w + xx;
             idx_start = (i-1)*Model.filterBetaSize+ Model.layercof_filters_startIDX + (layer-1)*Model.len_layercof;
             idx_end = idx_start+Model.filterBetaSize-1;
             cof_beta = Model.cof(idx_start : idx_end);
             tmpf = Model.Basis * cof_beta;
             filters2D = reshape(tmpf./norm(tmpf), Model.filterWidth, Model.filterWidth);       
             subaxis(vis_grid_h, vis_grid_w, i, 'Spacing', 0.025, 'Padding', 0, 'Margin', 0.025);
             imagesc(filters2D), colormap(gray), axis equal, axis off; 
         end
    end

    if(Training.isPreTraining)
       saveas(hFigure, sprintf('%s/filters_pretrain_iter_%d_layer_%d.png', result_dir, Training.iter_training, layer), 'png');      
    else
       saveas(hFigure, sprintf('%s/filters_iter_%d_layer_%d.png', result_dir, Training.iter_training, layer), 'png');      
    end
end

end