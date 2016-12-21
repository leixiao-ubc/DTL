function [] = visualizeInfluenceFunctions()

global Training Model result_dir

vis_grid_h = Model.filterWidth-1;
vis_grid_w = Model.filterWidth+1;

dnv = Training.data_normalization_value;
in = dnv.*[-1.2:0.01:1.2]; 
if(strcmp(Model.model_type, 'csf'))
    out_range = [-dnv, dnv];
else
    out_range = [-0.1*dnv, 0.1*dnv];
end
rbf_scale_constant = 1/(2*Model.RBFscale^2);

for layer = 1:Model.numDenoiseLayers
    hFigure = figure(1); close(hFigure);
    hFigure = figure(1); set(hFigure,'visible','off');
    for yy = 1:vis_grid_h
         for xx = 1:vis_grid_w
            idx= (yy - 1)*vis_grid_w + xx;
            out = zeros(size(in));                  
            if(Model.use_symmetric_gmm)
                start = (idx - 1)*Model.numRBFs_reduced + Model.layercof_RBFs_startIDX + (layer-1)*Model.len_layercof - 1; 
                for k = 1:Model.numRBFs_reduced 
                    out = out + Model.cof(start+k).* exp(-(in - Model.RBFcenters(k+Model.numRBFs_reduced+1)).^2*rbf_scale_constant);
                    out = out - Model.cof(start+k).* exp(-(in + Model.RBFcenters(k+Model.numRBFs_reduced+1)).^2*rbf_scale_constant);
                end
            else
                start = (idx - 1)*Model.numRBFs + Model.layercof_RBFs_startIDX + (layer-1)*Model.len_layercof - 1; 
                for k = 1:Model.numRBFs
                    out = out + Model.cof(start+k).* exp(-(in - Model.RBFcenters(k)).^2*rbf_scale_constant);
                end
            end
            subaxis(vis_grid_h, vis_grid_w, idx, 'Spacing', 0.035, 'Padding', 0, 'Margin', 0.035);
            plot(in, in, '--k', 'LineWidth', 0.5), hold on; axis on, axis([in(1) in(end) out_range(1) out_range(end)]); %, axis equal
            plot(in, out, '-r', 'LineWidth', 1), axis on, axis([in(1) in(end) out_range(1) out_range(end)]); %, axis equal
         end
    end

    if(Training.isPreTraining)
        saveas(hFigure, sprintf('%s/operator_pretrain_iter_%d_layer_%d.png', result_dir, Training.iter_training, layer), 'png');   
    else
        saveas(hFigure, sprintf('%s/operator_iter_%d_layer_%d.png', result_dir, Training.iter_training, layer), 'png');    
    end
end

end




   