
Multi_algo_HP_dict = {
    'IForest': {
        'n_estimators': [25, 50, 100, 150, 200],
        'max_features': [0.2, 0.4, 0.6, 0.8, 1.0]
    },
    'LOF': {
        'n_neighbors': [10, 20, 30, 40, 50],
        'metric': ['minkowski', 'manhattan', 'euclidean']
    },    
    'PCA': {
        'n_components': [0.25, 0.5, 0.75, None]
    },        
    'HBOS': {
        'n_bins': [5, 10, 20, 30, 40],
        'tol': [0.1, 0.3, 0.5, 0.7]
    },
    'OCSVM': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'nu': [0.1, 0.3, 0.5, 0.7]
    },        
    'MCD': {
        'support_fraction': [0.2, 0.4, 0.6, 0.8, None]
    },
    'KNN': {
        'n_neighbors': [10, 20, 30, 40, 50],
        'method': ['largest', 'mean', 'median']
    },        
    'KMeansAD': {
        'n_clusters': [10, 20, 30, 40],
        'window_size': [10, 20, 30, 40]
    },
    'COPOD': {
        'HP': [None]
    },    
    'CBLOF': {
        'n_clusters': [4, 8, 16, 32],
        'alpha': [0.6, 0.7, 0.8, 0.9]
    },
    'EIF': {
        'n_trees': [25, 50, 100, 200]
    },   
    'RobustPCA': {
        'max_iter': [500, 1000, 1500]
    },
    'AutoEncoder': {
        'hidden_neurons': [[64, 32], [32, 16], [128, 64]]
    },
    'CNN': {
        'window_size': [50, 100, 150],
        'num_channel': [[32, 32, 40], [16, 32, 64]]
    },
    'LSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.0004, 0.0008]
    },  
    'TranAD': {
        'win_size': [5, 10, 50],
        'lr': [1e-3, 1e-4]
    },  
    'AnomalyTransformer': {
        'win_size': [50, 100, 150],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'OmniAnomaly': {
        'win_size': [5, 50, 100],
        'lr': [0.002, 0.0002]
    },
    'USAD': {
        'win_size': [5, 50, 100],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'Donut': {
        'win_size': [60, 90, 120],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'TimesNet': {
        'win_size': [32, 96, 192],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'FITS': {
        'win_size': [100, 200],
        'lr': [1e-3, 1e-4, 1e-5]
    },    
    'OFA': {
        'win_size': [50, 100, 150]
    },
    'StreamVAE': {
        'win_size': [50, 100, 150],
        'latent_dim': [32, 64, 128],
        'lr': [1e-3, 5e-4, 1e-4],
        'target_kl': [50.0, 100.0, 200.0]
    },
    'AxonAD': {
        'win_size': [50, 100],
        'd_model': [64, 128],
        'num_heads': [4, 8],
        'lr': [5e-4, 1e-3],
        'kl_tail_k': [3, 5, 10],
        'forecast_steps': [1, 3, 5, 25]
    },
    'WVAE': {
        'win_size': [50, 100, 150],
        'latent_dim': [5, 10, 20],
        'lr': [1e-3, 5e-4],
        'noise_std': [0.5, 0.8, 1.0]
    },
    'VSVAE': {
        'win_size': [50, 100, 150],
        'latent_dim': [3, 5, 10],
        'attn_vec_size': [3, 5, 10],
        'lr': [1e-3, 5e-4]
    },
    'VASP': {
        'win_size': [50, 100, 150],
        'latent_dim': [8, 16, 32],
        'lr': [1e-3, 5e-4],
        'kl_weight': [0.3, 0.5, 0.7]
    },
    'TFTResidual': {
        'win_size': [50, 100, 150],
        'pred_len': [5, 10, 15],
        'd_model': [64, 128],
        'd_hidden': [128, 256],
        'n_heads': [4, 8],
        'num_attn_layers': [1, 2],
        'dropout': [0.1, 0.2],
        'lr': [1e-3, 5e-4],
        'use_time_covariates': [True],
        'time_num_freqs': [2, 3]
    },
    'SISVAE': {
        'win_size': [50, 100, 150],
        'latent_dim': [20, 40, 60],
        'hidden_dim': [100, 200],
        'lr': [1e-3, 5e-4],
        'smooth_weight': [0.3, 0.5, 0.7]
    },
    'MAVAE': {
        'win_size': [50, 100],
        'latent_dim': [32, 64, 128],
        'n_heads': [4, 8],
        'lr': [1e-3, 5e-4],
        'noise_std': [0.01, 0.05]
    },
    'GDN': {
        'win_size': [50, 100, 150],
        'hidden_dim': [64, 128],
        'layers': [2, 3],
        'topk': [10, 15, 20],
        'lr': [1e-3, 5e-4]
}
}


Optimal_Multi_algo_HP_dict = {
    'IForest': {'n_estimators': 25, 'max_features': 0.8},
    'LOF': {'n_neighbors': 50, 'metric': 'euclidean'},    
    'PCA': {'n_components': 0.25},        
    'HBOS': {'n_bins': 30, 'tol': 0.5},
    'OCSVM': {'kernel': 'rbf', 'nu': 0.1},        
    'MCD': {'support_fraction': 0.8},
    'KNN': {'n_neighbors': 50, 'method': 'mean'},        
    'KMeansAD': {'n_clusters': 10, 'window_size': 40},
    'KShapeAD': {'n_clusters': 20, 'window_size': 40},
    'COPOD': {'n_jobs':1},    
    'CBLOF': {'n_clusters': 4, 'alpha': 0.6},
    'EIF': {'n_trees': 50},   
    'RobustPCA': {'max_iter': 1000},
    'AutoEncoder': {'hidden_neurons': [128, 64]},
    'CNN': {'window_size': 50, 'num_channel': [32, 32, 40]},
    'LSTMAD': {'window_size': 150, 'lr': 0.0008},  
    'TranAD': {'win_size': 10, 'lr': 0.001},  
    'AnomalyTransformer': {'win_size': 50, 'lr': 0.001},  
    'OmniAnomaly': {'win_size': 100, 'lr': 0.002},
    'USAD': {'win_size': 100, 'lr': 0.001},  
    'Donut': {'win_size': 60, 'lr': 0.001},  
    'TimesNet': {'win_size': 96, 'lr': 0.0001},
    'FITS': {'win_size': 100, 'lr': 0.001},
    'OFA': {'win_size': 50},
    'StreamVAE': {'win_size': 100, 'latent_dim': 64, 'batch_size': 128, 'epochs': 50, 'lr': 0.001, 'target_kl': 100.0},
    'AxonAD': {'win_size': 100, 'd_model': 128, 'num_heads': 8, 'lr': 0.0005, 'kl_tail_k': 10, 'forecast_steps': 1},
    'WVAE': {'win_size': 100, 'latent_dim': 10, 'lr': 0.001, 'noise_std': 0.8},
    'VSVAE': {'win_size': 100, 'latent_dim': 5, 'attn_vec_size': 5, 'lr': 0.001},
    'VASP': {'win_size': 100, 'latent_dim': 16, 'lr': 0.001, 'kl_weight': 0.5},
    'TFTResidual': {'win_size': 100, 'pred_len': 10, 'd_model': 64, 'd_hidden': 128, 'n_heads': 4, 'num_attn_layers': 1, 'dropout': 0.1, 'lr': 0.001, 'use_time_covariates': True, 'time_num_freqs': 2},
    'SISVAE': {'win_size': 100, 'latent_dim': 40, 'hidden_dim': 200, 'lr': 0.001, 'smooth_weight': 0.5},
    'MAVAE': {'win_size': 100, 'latent_dim': 64, 'n_heads': 8, 'lr': 0.001, 'noise_std': 0.01},
    'GDN': {'win_size': 100, 'hidden_dim': 128, 'layers': 2, 'topk': 15, 'lr': 0.001}
}


Uni_algo_HP_dict = {
    'Sub_IForest': {
        'periodicity': [1, 2, 3],
        'n_estimators': [25, 50, 100, 150, 200]
    },
    'IForest': {
        'n_estimators': [25, 50, 100, 150, 200]
    },
    'Sub_LOF': {
        'periodicity': [1, 2, 3],
        'n_neighbors': [10, 20, 30, 40, 50]
    }, 
    'LOF': {
        'n_neighbors': [10, 20, 30, 40, 50]
    }, 
    'POLY': {
        'periodicity': [1, 2, 3],
        'power': [1, 2, 3, 4]
    },
    'MatrixProfile': {
        'periodicity': [1, 2, 3]
    },
    'NORMA': {
        'periodicity': [1, 2, 3],
        'clustering': ['hierarchical', 'kshape']
    },
    'SAND': {
        'periodicity': [1, 2, 3]
    }, 
    'Series2Graph': {
        'periodicity': [1, 2, 3]
    },
    'Sub_PCA': {
        'periodicity': [1, 2, 3],
        'n_components': [0.25, 0.5, 0.75, None]
    },
    'Sub_HBOS': {
        'periodicity': [1, 2, 3],
        'n_bins': [5, 10, 20, 30, 40]
    },
    'Sub_OCSVM': {
        'periodicity': [1, 2, 3],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    },
    'Sub_MCD': {
        'periodicity': [1, 2, 3],
        'support_fraction': [0.2, 0.4, 0.6, 0.8, None]
    },
    'Sub_KNN': {
        'periodicity': [1, 2, 3],
        'n_neighbors': [10, 20, 30, 40, 50],
    },
    'KMeansAD_U': {
        'periodicity': [1, 2, 3],
        'n_clusters': [10, 20, 30, 40],
    },
    'KShapeAD': {
        'periodicity': [1, 2, 3]
    },
    'AutoEncoder': {
        'window_size': [50, 100, 150],
        'hidden_neurons': [[64, 32], [32, 16], [128, 64]]
    },
    'CNN': {
        'window_size': [50, 100, 150],
        'num_channel': [[32, 32, 40], [16, 32, 64]]
    },
    'LSTMAD': {
        'window_size': [50, 100, 150],
        'lr': [0.0004, 0.0008]
    },  
    'TranAD': {
        'win_size': [5, 10, 50],
        'lr': [1e-3, 1e-4]
    },
    'AnomalyTransformer': {
        'win_size': [50, 100, 150],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'OmniAnomaly': {
        'win_size': [5, 50, 100],
        'lr': [0.002, 0.0002]
    },
    'USAD': {
        'win_size': [5, 50, 100],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'Donut': {
        'win_size': [60, 90, 120],
        'lr': [1e-3, 1e-4, 1e-5]
    },  
    'TimesNet': {
        'win_size': [32, 96, 192],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'FITS': {
        'win_size': [100, 200],
        'lr': [1e-3, 1e-4, 1e-5]
    },
    'OFA': {
        'win_size': [50, 100, 150]
    },    
    'Lag_Llama': {
        'win_size': [32, 64, 96]
    },    
    'Chronos': {
        'win_size': [50, 100, 150]
    },
    'TimesFM': {
        'win_size': [32, 64, 96]
    },
    'MOMENT_ZS': {
        'win_size': [64, 128, 256]
    },
    'MOMENT_FT': {
        'win_size': [64, 128, 256]
    }
}

Optimal_Uni_algo_HP_dict = {
    'Sub_IForest': {'periodicity': 1, 'n_estimators': 150},
    'IForest': {'n_estimators': 200},
    'Sub_LOF': {'periodicity': 2, 'n_neighbors': 30},
    'LOF': {'n_neighbors': 50},
    'POLY': {'periodicity': 1, 'power': 4},
    'MatrixProfile': {'periodicity': 1},
    'NORMA': {'periodicity': 1, 'clustering': 'kshape'},
    'SAND': {'periodicity': 1},
    'Series2Graph': {'periodicity': 1},
    'SR': {'periodicity': 1},
    'Sub_PCA': {'periodicity': 1, 'n_components': None},        
    'Sub_HBOS': {'periodicity': 1, 'n_bins': 10},
    'Sub_OCSVM': {'periodicity': 2, 'kernel': 'rbf'},        
    'Sub_MCD': {'periodicity': 3, 'support_fraction': None},
    'Sub_KNN': {'periodicity': 2, 'n_neighbors': 50}, 
    'KMeansAD_U': {'periodicity': 2, 'n_clusters': 10},
    'KShapeAD': {'periodicity': 1},
    'FFT': {},
    'Left_STAMPi': {},
    'AutoEncoder': {'window_size': 100, 'hidden_neurons': [128, 64]},
    'CNN': {'window_size': 50, 'num_channel': [32, 32, 40]},
    'LSTMAD': {'window_size': 100, 'lr': 0.0008},  
    'TranAD': {'win_size': 10, 'lr': 0.0001},
    'AnomalyTransformer': {'win_size': 50, 'lr': 0.001},  
    'OmniAnomaly': {'win_size': 5, 'lr': 0.002},
    'USAD': {'win_size': 100, 'lr': 0.001},
    'Donut': {'win_size': 60, 'lr': 0.0001},  
    'TimesNet': {'win_size': 32, 'lr': 0.0001},
    'FITS': {'win_size': 100, 'lr': 0.0001},
    'OFA': {'win_size': 50},
    'Lag_Llama': {'win_size': 96},
    'Chronos': {'win_size': 100},
    'TimesFM': {'win_size': 96},
    'MOMENT_ZS': {'win_size': 64},
    'MOMENT_FT': {'win_size': 64},
    'M2N2': {}
}