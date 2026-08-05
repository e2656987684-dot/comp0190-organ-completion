"""
The published MSN demo architecture, lifted VERBATIM out of
notebooks/demo/MSN_model_inference_demo.ipynb (cells 4/6/8/10/12).

WHY THIS FILE EXISTS -- the weights only fit this exact topology.

`msn_skullfix.py` is a rewrite of the same network, and its docstring claims
`paper()` can load msn_downloads/MSN_weights3.h5. That is wrong, and it fails
SILENTLY, which is worse: `load_weights(..., by_name=True, skip_mismatch=True)`
returns without raising while matching only 3 of the 32 stored weight groups
(`D-OUT_lin`, `D1-IN`, `D2-IN`) -- i.e. ~96% of the network stays randomly
initialised and you get garbage predictions with no error.

The cause is layer nesting, not shapes. The demo's `LBR` wraps its Dense+ReLU
in a nested keras Model named e.g. `E-IN_LBR1`, so the checkpoint stores
`E-IN_LBR1/E-IN_LBR1_lin/kernel`. The rewrite builds a bare Dense named
`E-IN_LBR1_lin` at the top level, so it looks for `E-IN_LBR1_lin/kernel`. Same
maths, same shapes, different paths -- by_name matching cannot bridge it. The
attention blocks differ the same way (`E-SA1` as one group vs `E-SA1_Q/_K/_V`).

So: use THIS module to run the published weights, and `msn_skullfix.py` for
models trained by this project. Do not try to cross-load between them.

Do not "clean up" the code below -- every layer name here is load-bearing.
"""

import os

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers as L
from tensorflow.keras import models as M
from transformers import TFBertModel, BertTokenizer

def pairwise_distance(xyz1, xyz2):
    n = xyz1.shape[1]
    c = xyz1.shape[2]
    m = xyz2.shape[1]
    # 这里会 materialize 一个 (N, M, 3) 的两两距离张量（比如 4096x4096x3），
    # 换过 cuda_malloc_async 分配器还是会 OOM，说明确实是显存不够而不是碎片化，
    # 挪到 CPU 算（这几千个点的规模对 CPU 来说很快），不影响模型输出结果。
    with tf.device('/CPU:0'):
        xyz1 = tf.tile(tf.reshape(xyz1, (-1,1,n,c)), [1,m,1,1])
        xyz2 = tf.tile(tf.reshape(xyz2, (-1,m,1,c)), [1,1,n,1])
        dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    return dist

def knn_point(k, xyz1, xyz2):
    dist = -pairwise_distance(xyz1, xyz2)
    val, idx = tf.math.top_k(dist, k)
    return -val, idx

class UniformSampler(tf.keras.layers.Layer):
    def __init__(self, num_points, seed=42, **kwargs):
        super(UniformSampler, self).__init__(**kwargs)
        self.num_points = num_points
        self.seed = seed

    def build(self, input_shape):
        pass

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        data_size = tf.shape(inputs)[1]
        indices = tf.random.uniform(
            shape=(batch_size, self.num_points),
            minval=0,
            maxval=data_size,
            dtype=tf.int32,
            seed=self.seed
        )
        return indices

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_points, input_shape[2])

    def get_config(self):
        config = super(UniformSampler, self).get_config()
        config.update({
            "num_points": self.num_points,
            "seed": self.seed
        })
        return config

def sample_and_group(args, nsample):
    xyz, pts, fps_idx = args
    new_xyz = tf.gather_nd(xyz, tf.expand_dims(fps_idx,-1), batch_dims=1)
    new_pts = tf.gather_nd(pts, tf.expand_dims(fps_idx,-1), batch_dims=1)
    _, idx = knn_point(nsample, xyz, new_xyz)
    grouped_pts = tf.gather_nd(pts, tf.expand_dims(idx,-1), batch_dims=1)
    grouped_pts -= tf.tile(tf.expand_dims(new_pts, 2),
                           (1,1,nsample,1))
    new_pts = tf.concat([grouped_pts,
                         tf.tile(tf.expand_dims(new_pts, 2),
                                 (1,1,nsample,1))],
                        axis=-1)
    return new_xyz, new_pts

def LBR(tensor, C, seq_name, use_bias=True, activation=None, LeakyAlpha=0.0):
    x_in = Input(shape=tensor.shape[1:], name=seq_name+'_input')
    x = L.Dense(C, use_bias=use_bias, activation=activation, name=seq_name+'_lin')(x_in)
    if LeakyAlpha==0.0:
        x_out = L.ReLU(name=seq_name+'_ReLU')(x)
    else:
        x_out = L.LeakyReLU(alpha=LeakyAlpha, name=seq_name+'_ReLU')(x)
    model = M.Model(inputs=x_in, outputs=x_out, name=seq_name)
    return model(tensor)

def Self_Attention(tensor, seq_name):
    x_in = Input(shape=tensor.shape[1:], name=seq_name+'_input')
    C = x_in.shape[2]
    W_q = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_Q')
    W_k = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_K')
    W_v = L.Dense(C, use_bias=False, activation=None, name=seq_name+'_V')
    x_q = W_q(x_in)
    x_k = W_k(x_in)
    W_k.set_weights(W_q.get_weights())
    x_k = L.Lambda(lambda t: tf.transpose(t, perm=(0,2,1)), name=seq_name+'_KT')(x_k)
    x_v = W_v(x_in)
    energy = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul1')([x_q, x_k])
    attention = L.Softmax(axis=1, name=seq_name+'_softmax')(energy)
    attention = L.Lambda(lambda t: t / (1e-9 + tf.reduce_sum(t, axis=2, keepdims=True)), name=seq_name+'_l1norm')(attention)
    x_r = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul2')([attention, x_v])
    x_r = L.Lambda(lambda ts: tf.subtract(ts[0],ts[1]), name=seq_name+'_subtract')([x_in, x_r])
    x_r = LBR(x_r, C, seq_name+'_LBR', use_bias=True)
    x_out = L.Lambda(lambda ts: tf.add(ts[0],ts[1]), name=seq_name+'_add')([x_in, x_r])
    model = M.Model(inputs=x_in, outputs=x_out, name=seq_name)
    return model(tensor)


def Cross_Attention(args, seq_name):
    E_tensor, D_tensor = args
    xE_in = Input(shape=E_tensor.shape[1:], name=seq_name+'_input-E')
    C = xE_in.shape[2]
    xD_in = Input(shape=D_tensor.shape[1:], name=seq_name+'_input-D')
    out_dim = xD_in.shape[2]
    W_q = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_Q')
    W_k = L.Dense(C//4, use_bias=False, activation=None, name=seq_name+'_K')
    W_v = L.Dense(out_dim, use_bias=False, activation=None, name=seq_name+'_V')
    x_q = W_q(xD_in)
    x_k = W_k(xE_in)
    x_k = L.Lambda(lambda t: tf.transpose(t, perm=(0,2,1)), name=seq_name+'_KT')(x_k)
    x_v = W_v(xE_in)
    energy = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul1')([x_q, x_k])
    attention = L.Softmax(axis=1, name=seq_name+'_softmax')(energy)
    attention = L.Lambda(lambda t: t / (1e-9 + tf.reduce_sum(t, axis=2, keepdims=True)), name=seq_name+'_l1norm')(attention)
    x_r = L.Lambda(lambda ts: tf.matmul(ts[0],ts[1]), name=seq_name+'_matmul2')([attention, x_v])
    x_r = L.Lambda(lambda ts: tf.subtract(ts[0],ts[1]), name=seq_name+'_subtract')([xD_in, x_r])
    x_r = LBR(x_r, out_dim, seq_name+'_LBR', use_bias=True)
    x_out = L.Lambda(lambda ts: tf.add(ts[0],ts[1]), name=seq_name+'_add')([xD_in, x_r])
    model = M.Model(inputs=[xE_in,xD_in], outputs=x_out, name=seq_name)
    return model([E_tensor,D_tensor])

def copy_and_mapping(tensor, nmul, seq_name):
    x_in = Input(shape=tensor.shape[1:], name=seq_name+'_input')
    x = L.Lambda(lambda t: tf.expand_dims(t, 2), name=seq_name+'_expand')(x_in)
    C = x.shape[-1]//nmul
    x1 = L.Conv2DTranspose(C,(1,nmul),(1,nmul), use_bias=True, activation=None, name=seq_name+'_convT')(x)
    x2 = L.Dense(C, use_bias=True, activation=None, name=seq_name+'_lin')(x)
    x2 = L.Lambda(lambda t: tf.tile(t, [1,1,nmul,1]), name=seq_name+'_tile')(x2)
    x = L.Lambda(lambda ts: tf.add(ts[0],ts[1]), name=seq_name+'_add')([x1, x2])
    npoint = x.shape[1]*x.shape[2]
    x_out = L.Lambda(lambda t: tf.reshape(t, [-1,npoint,t.shape[3]]), name=seq_name+'_reshape')(x)
    model = M.Model(inputs=x_in, outputs=x_out, name=seq_name)
    return model(tensor)

def PCT_encoder(xyz):
    x = LBR(xyz, 64, 'E-IN_LBR1', use_bias=False)
    x = LBR(x, 128, 'E-IN_LBR2', use_bias=False)
    fps_idx = UniformSampler(4096)(xyz)
    new_xyz, new_feature = L.Lambda(sample_and_group, arguments={'nsample':32}, name='E-SG1')([xyz, x, fps_idx])
    x = LBR(new_feature, 512, 'E-SG1_LBR1', use_bias=False)
    x = L.Lambda(lambda t: tf.reduce_max(t, axis=2), name='E-SG1_MaxPool')(x)
    fps_idx = UniformSampler(2048)(new_xyz)
    new_xyz, new_feature = L.Lambda(sample_and_group, arguments={'nsample':32}, name='E-SG2')([new_xyz, x, fps_idx])
    x = LBR(new_feature, 1024, 'E-SG2_LBR1', use_bias=False)
    x = L.Lambda(lambda t: tf.reduce_max(t, axis=2), name='E-SG2_MaxPool')(x)
    x1 = Self_Attention(x, 'E-SA1')
    x2 = Self_Attention(x1, 'E-SA2')
    x3 = Self_Attention(x2, 'E-SA3')
    x4 = Self_Attention(x3, 'E-SA4')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='E-SA_Concat')([x1,x2,x3,x4])
    x = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='E-OUT_Concat')([x0,x])
    x = LBR(x, 2048, 'E-OUT_LBR', use_bias=False, LeakyAlpha=0.2)
    x1 = Self_Attention(x, 'E-SA5')
    x2 = Self_Attention(x1, 'E-SA6')
    x3 = Self_Attention(x2, 'E-SA7')
    x4 = Self_Attention(x3, 'E-SA8')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='E-SA_Concat2')([x1,x2,x3,x4])
    x = LBR(x0, 4096, 'E-OUT_LBR1', use_bias=False, LeakyAlpha=0.2)
    output_feats = L.Lambda(lambda t: tf.reduce_max(t, axis=1, keepdims=True), name='E-OUT_MaxPool')(x)
    return output_feats

def pct_decoder(input_feats, input_eye_seed):
    m_feats = L.Lambda(lambda x: tf.tile(x, [1,1024,1]), name = 'D-IN_replicate')(input_feats)
    input_eye = input_eye_seed + tf.eye(1024,1024)
    x = L.Dense(4096//4, use_bias=False, activation=None, name='D1-IN')(input_eye)
    x1 = Cross_Attention([m_feats,x] , 'D-STA1')
    x2 = Cross_Attention([m_feats,x1], 'D-STA2')
    x3 = Cross_Attention([m_feats,x2], 'D-STA3')
    x4 = Cross_Attention([m_feats,x3], 'D-STA4')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D1-STA_Concat')([x1,x2,x3,x4])
    x = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D1-OUT_Concat')([x0,x])
    m_feats2 = copy_and_mapping(x, 3, 'D1-OUT_CopyAndMapping')
    input_eye2 = input_eye_seed + tf.eye(3072,3072)
    x = L.Dense(1024//4, use_bias=False, activation=None, name='D2-IN')(input_eye2)
    x1 = Cross_Attention([m_feats2,x] , 'D2-STA1')
    x2 = Cross_Attention([m_feats2,x1], 'D2-STA2')
    x3 = Cross_Attention([m_feats2,x2], 'D2-STA3')
    x4 = Cross_Attention([m_feats2,x3], 'D2-STA4')
    x0 = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D2-STA_Concat')([x1,x2,x3,x4])
    x = L.Lambda(lambda ts: tf.concat(ts, axis=2), name='D2-OUT_Concat')([x0,x])
    x = copy_and_mapping(x, 2, 'D2-OUT_CopyAndMapping')
    x = LBR(x,128, 'D-OUT_LBR1', use_bias=False)
    x = LBR(x,128, 'D-OUT_LBR2', use_bias=False)
    x = LBR(x, 64, 'D-OUT_LBR3', use_bias=False, LeakyAlpha=0.2)
    output_points = L.Dense(3, activation=None, name='D-OUT_lin')(x)
    return output_points

def bert_model(input_ids, attention_mask, model_name='bert-base-uncased', max_length=128):
    bert_model = TFBertModel.from_pretrained(model_name, use_safetensors=False)
    bert_model.trainable = False
    bert_outputs = bert_model([input_ids, attention_mask])
    cls_output = bert_outputs.pooler_output
    dense_output = tf.keras.layers.Dense(4096, activation='relu')(cls_output)
    output = tf.expand_dims(dense_output, axis = 1)
    return output

class PCT_AE_Multimodal:
    def __init__(self, num_input_points=4096, max_length=128, bert_model=bert_model, PCT_encoder=PCT_encoder, pct_decoder=pct_decoder):
        self.num_input_points = num_input_points
        self.max_length = max_length
        self.bert_model = bert_model
        self.PCT_encoder = PCT_encoder
        self.pct_decoder = pct_decoder
        self.model = self.build_model()

    def build_model(self):
        eye_seed = Input(shape=(1, 1), name='input_eye_seed')
        xyz = Input(shape=(self.num_input_points, 3), name='input_points')
        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
        if not self.bert_model or not self.PCT_encoder or not self.pct_decoder:
            raise ValueError("Bert model, PCT encoder, and PCT decoder must be provided.")
        text_encoded = self.bert_model(input_ids, attention_mask)
        cloud_encoded = self.PCT_encoder(xyz)
        multi_encoded = cloud_encoded + text_encoded
        output = self.pct_decoder(multi_encoded, eye_seed)
        return M.Model(inputs=[xyz, eye_seed, input_ids, attention_mask], outputs=output)
