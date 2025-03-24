from torch.utils.data import DataLoader
import warnings
from utils import load_data
import json
from layers import Agent
from utils import make_dirs,write_data
import torch.optim.lr_scheduler as lr_scheduler
from data_utils import *
from conf import *
warnings.filterwarnings("ignore")
# load dataset
train_s, test_s = load_data(train_path), load_data(test_path)
record = {}
train_samples = train_s
test_samples = test_s

train_size, test_size = len(train_samples), len(test_samples)

sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size)
dv = DiseaseVocab(samples=train_samples)

num_sxs, num_dis = sv.num_sxs, dv.num_dis

train_ds = SymptomDataset(train_samples, sv, dv, keep_unk=False, add_tgt_start=True, add_tgt_end=True)
train_ds_loader = DataLoader(train_ds, batch_size=train_bsz, num_workers=num_workers, shuffle=True, collate_fn=lm_collater)
notrain_ds_loader = DataLoader(train_ds, batch_size=train_size, num_workers=num_workers, shuffle=False, collate_fn=lm_collater)

test_ds = SymptomDataset(test_samples, sv, dv, keep_unk=False, add_tgt_start=True, add_tgt_end=True)
test_ds_loader = DataLoader(test_ds, batch_size=test_size, num_workers=num_workers, shuffle=False, collate_fn=lm_collater)

dc_criterion = torch.nn.CrossEntropyLoss().to(device)
dc_criterion_2 = torch.nn.CrossEntropyLoss(reduction='none').to(device)
make_dirs([best_pt_path, last_pt_path])

print('training...')
k_dict = {}
for k in np.arange(0.01, 0.5, 0.01):

    acc_h = []
    acc_m = []
    acc_hm = []
    acc_final = []
    utility = []
    ut_record = []
    ad_per_error = []
    ad_per_all = []
    h_err, final_err = [], []

    for i in range(5):
        best_utility = 0
        bset_acc_h = 0
        bset_acc_m = 0
        bset_acc_hm = 0
        bset_acc_final = 0
        set_seed(i+1)
        model = Agent(num_sxs, num_dis, sv).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=pt_learning_rate)
        human = HumanAgent(notrain_ds_loader, test_ds_loader)
        for epoch in range(pt_train_epochs):
            train_num_hits = 0
            model.train()
            for batch in train_ds_loader:
                sx_ids, attr_ids, labels, ids = batch['sx_ids'], batch['attr_ids'], batch['labels'], batch['ids']
                si_sx_feats, si_attr_feats = make_features_xfmr(
                    sv, batch, sx_ids.permute(1, 0), attr_ids.permute(1, 0), merge_act=False, merge_si=True)
                dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                h_preds = human.tr_pred[ids]
                norm_dc_outputs = torch.softmax(dc_outputs, dim=-1)
                hm_outputs = human.combine_hm(norm_dc_outputs, h_preds)
                loss = dc_criterion(dc_outputs, batch['labels'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval() 
            for batch in test_ds_loader:
                sx_ids, attr_ids, labels, ids = batch['sx_ids'], batch['attr_ids'], batch['labels'], batch['ids']
                si_sx_feats, si_attr_feats = make_features_xfmr(
                    sv, batch, sx_ids.permute(1, 0), attr_ids.permute(1, 0), merge_act=False, merge_si=True)
                dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                h_preds = human.te_pred[ids]
                norm_dc_outputs = torch.softmax(dc_outputs, dim=-1)
                hm_outputs = human.combine_hm(norm_dc_outputs, h_preds)
                # Metrics
                dispersed_utility, final_preds, nums_A_R_T_F, advice_all, advice_label = human.compute_dispersed_utility(h_preds, norm_dc_outputs, labels)
                test_num_h = torch.sum(batch['labels'].eq(h_preds)).item()
                test_num_m = torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()
                test_num_hm = torch.sum(batch['labels'].eq(hm_outputs.argmax(dim=-1))).item()
                test_num_final = torch.sum(batch['labels'].eq(final_preds)).item()
                if print_per == True :
                    advice_per_error, advice_per_all = human.print_acc_per_class(advice_label, advice_all)
                    human_per_error, final_per_error = human.print_error_per_class(labels, final_preds, h_preds)
            test_acc_h = test_num_h / test_size
            test_acc_m = test_num_m / test_size
            test_acc_hm = test_num_hm / test_size
            test_acc_final = test_num_final / test_size
            # checkpoint
            if test_acc_final > bset_acc_final:
                best_utility = dispersed_utility
                bset_acc_h = test_acc_h
                bset_acc_m = test_acc_m
                bset_acc_hm = test_acc_hm
                bset_acc_final = test_acc_final
                test_record = nums_A_R_T_F
                if print_per == True :
                    best_ad_error = advice_per_error
                    best_ad_all = advice_per_all
                    best_h_error = human_per_error
                    best_final_error = final_per_error
                # print(dispersed_utility)
                # print(test_acc_final)
        acc_h.append(test_acc_h)
        acc_m.append(bset_acc_m)
        acc_hm.append(bset_acc_hm)
        acc_final.append(bset_acc_final)
        utility.append(best_utility)
        ut_record.append(test_record)
        if print_per == True:
            ad_per_error.append(best_ad_error)
            ad_per_all.append(best_ad_all)
            h_err.append(best_h_error)
            final_err.append(best_final_error)
        print(acc_final)
    print("-" * 50)
    # print('acc_h: {}'.format(round(sum(acc_h) / len(acc_h), digits)))
    print('acc_m: {}'.format(round(sum(acc_m) / len(acc_m), digits)))
    # print('acc_hm: {}'.format(round(sum(acc_hm) / len(acc_hm), digits)))
    print('dispersed_utility: {}'.format(round(sum(utility) / len(utility), digits)))
    print('acc_final: {}'.format(round(sum(acc_final) / len(acc_final), digits)))
    print(list(np.mean(np.array(ut_record), axis =0)))
    if print_per == True:
        print([round(i, digits)for i in list(np.mean(np.array(h_err), axis =0))])
        print([round(i, digits)for i in list(np.mean(np.array(final_err), axis =0))])
        print([round(i, digits)for i in list(np.mean(np.array(ad_per_error), axis =0))])
        print([round(i, digits)for i in list(np.mean(np.array(ad_per_all), axis =0))])
    assert 0
    k_dict[str(k)] = format(round(sum(utility) / len(utility), digits))
    print(k_dict)
        

