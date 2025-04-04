import torch
import torch.nn.functional as F
from torchvision import transforms
from pi_seg.inference.transforms import SigmoidForPred


class PI_Predictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 infer_size = 256,
                 focus_crop_r = 1.4,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None
        self.uncertainty_map = None
        self.prev_mask = None
        
        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()
        self.transforms = []
        self.crop_l = infer_size
        self.focus_crop_r = focus_crop_r
        self.transforms.append(SigmoidForPred())
        self.focus_roi = None 
        self.global_roi = None 


    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])

    def set_prev_mask(self, mask):
        self.prev_prediction = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()

    def get_prediction(self, clicker, sample_cnt=16, prev_mask=None):
        clicks_list = clicker.get_clicks()
        click = clicks_list[-1]
        last_y,last_x = click.coords[0],click.coords[1]
        self.last_y = last_y
        self.last_x = last_x

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net.module, 'with_prev_mask') and self.net.module.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
           
        try:
            roi = self.transforms[0]._object_roi
            y1,y2,x1,x2 = roi
            global_roi = (y1,y2+1,x1,x2+1)
        except:
            h,w = prev_mask.shape[-2], prev_mask.shape[-1]
            global_roi = (0,h,0,w)
            
        self.global_roi = global_roi
    
        # prediction
        pred_logits, ps = self._get_prediction(image_nd, clicks_lists, sample_cnt, is_image_changed)
        coarse_mask = pred_logits
        
        clicks_list = clicker.get_clicks()
        image_full = self.original_image
        
        self.prev_prediction = coarse_mask
        self.transforms[0]._prev_probs = coarse_mask.cpu().numpy()
        
        return coarse_mask.cpu().numpy()[0, 0], self.uncertainty_map, ps

    def _get_prediction(self, image_nd, clicks_lists, sample_cnt, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        output = self.net.forward(image=image_nd, points=points_nd,
                                  sample_cnt=sample_cnt, training=False)

        ps = output['samples']
        ps = [torch.sigmoid(s) for s in ps]

        pm = torch.stack(ps).mean(dim=0)
        psd = torch.stack(ps).std(dim=0, unbiased=False)

        self.uncertainty_map = psd.to('cpu').detach().numpy()
        
        return pm, ps
    
    def mapp_roi(self, focus_roi, global_roi):
        yg1, yg2, xg1, xg2 = global_roi
        hg, wg = yg2-yg1, xg2-xg1
        yf1, yf2, xf1, xf2 = focus_roi
        
        yf1_n = (yf1-yg1) * (self.crop_l/hg)
        yf2_n = (yf2-yg1) * (self.crop_l/hg)
        xf1_n = (xf1-xg1) * (self.crop_l/wg)
        xf2_n = (xf2-xg1) * (self.crop_l/wg)

        yf1_n = max(yf1_n,0)
        yf2_n = min(yf2_n,self.crop_l)
        xf1_n = max(xf1_n,0)
        xf2_n = min(xf2_n,self.crop_l)
        return (yf1_n, yf2_n, xf1_n, xf2_n)

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)
        print('_set_transform_states')

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        print('set')
