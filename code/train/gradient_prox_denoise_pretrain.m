function [g_cof_denoise] = gradient_prox_denoise_pretrain(cof, split_z_init, a, rho, Training, Model, split_z_local, phi, d_phi, c)
% compute gradient of objective w.r.t. model coefficients at denoise layers

currentLayer = Training.PreTrain.layer;

g_cof_denoise = zeros(Model.len_layercof, Model.numDenoiseLayers);

rsf = @(z) reshape(z, [Model.filterWidth, Model.filterWidth]); % function handle to reshape model filter from vector

layerscof = reshape(cof(1:end-1), Model.len_layercof, Model.numDenoiseLayers); % discard lambda

% backward, compute gradient
g_beta = zeros(Model.filterBetaSize*Model.numFilters, Model.numDenoiseLayers);
g_rbf = zeros(Model.numRBFs_trained*Model.numFilters, Model.numDenoiseLayers);
    
% compute q^i (i.e., g_split_z^i))
g_split_z_local = zeros(Training.imdims(1), Training.imdims(2), Model.numDenoiseLayers); 
    

%for i = Model.numDenoiseLayers:-1:1
for i = currentLayer:currentLayer
% gradient g >> split_z^(k) >> split_z^(k-1)    [to accumulate]  
% gradient g >> x^(t) >> alpha  [to accumulate]
% gradient g >> x^(t) >> f_i      [to accumulate]
% gradient g >> x^(t) >> RBF_i      [to accumulate]
% gradient g >> split_u^(t) >> alpha  [to accumulate]
% gradient g >> split_u^(t) >> f_i      [to accumulate]
% gradient g >> split_u^(t) >> RBF_i      [to accumulate]

    g_split_z_local(:,:,i) = rho.*a;
    
    q = g_split_z_local(:,:,i); 
    clique_q = q(Training.cliques_of_circ_conv) ;
    
    if(i>1)
        split_z_old = split_z_local(:,:,i-1);
    else
        split_z_old = split_z_init;
    end
    clique_split_z_old = split_z_old(Training.cliques_of_circ_conv);
        
    for j = 1:Model.numFilters
        % compute g>>...>>f_i
        filter = Model.filters2D(:,:,j,i);
        fq = imfilter_warp(q, filter);
        pj = phi(:,:,j,i);
        tp1 = -clique_q*pj(:);
        dpj = d_phi(:,:,j,i);
        tp2 = -clique_split_z_old*(dpj(:).*fq(:));
        g_filter = rsf(tp1+tp2);
        % then compute gradient w.r.t. filter coefficients
        idx_start = (j-1)*Model.filterBetaSize+Model.layercof_filters_startIDX;
        idx_end = idx_start+Model.filterBetaSize-1;
        cof_beta = layerscof(idx_start : idx_end, i);
        cof_beta_t = cof_beta';
        r = (cof_beta_t*Model.Basis_t_Basis*cof_beta).^(-1/2); % scalar   
        g_filter_beta = r.*Model.Basis_t - r.^3.*Model.Basis_t_Basis*(cof_beta*cof_beta_t)*Model.Basis_t; % this is a matrix
        g_beta(idx_start : idx_end,i) = g_filter_beta*g_filter(:);
        
        % gradient g >>...>> RBF_i  [to accumulate]
        idx = (j-1)*Model.numRBFs_trained; 
        for r = 1:Model.numRBFs_trained 
             c_local = c(r,:,j,i);
             g_rbf(idx+r, i) = g_rbf(idx+r, i) + dot(c_local(:), -fq(:));  
        end           
    end
    
    g_cof_denoise(:, i) = [g_beta(:,i); g_rbf(:,i)];
    
end

g_cof_denoise = g_cof_denoise(:); % vectorize


end