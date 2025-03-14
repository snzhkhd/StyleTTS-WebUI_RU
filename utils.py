from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
import random
import phonemizer

from nltk.tokenize import word_tokenize
from munch import Munch
from styletts2.models import *
from styletts2.utils import *
from styletts2.text_utils import TextCleaner
from styletts2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from ruphon import RUPhon
from ruaccent import RUAccent

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    
    
    
def set_seeds(seed=0):
    if seed==-1:
        seed_value = random.randint(0, 2**32 - 1)
    else:
        seed_value = seed
    torch.manual_seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed_value)
    np.random.seed(seed_value)
    
def load_configurations(config_path):
    return yaml.safe_load(open(config_path))

def load_models(config, device):
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = config.get('PLBERT_dir', False)
    from Utils.PLBERT.util import load_plbert
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    
    return model, model_params

def load_pretrained_model(model, model_path):
    params_whole = torch.load(model_path, map_location='cpu')
    params = params_whole['net']
    for key in model:
        if key in params:
            print(f'{key} loaded')
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]
    
def create_sampler(model):
    return DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

def preprocess(wave, to_mel, mean, std):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, model, to_mel, mean, std, device):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio, to_mel, mean, std).to(device)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)

def load_phonemizer():
    #global_phonemizer = phonemizer.backend.EspeakBackend(language='ru', preserve_punctuation=True, with_stress=True, language_switch='remove-flags')
    global_phonemizer = RUPhon()
    global_phonemizer.load("big", workdir="./models", device="cuda")
    global_accentizer = RUAccent()
    global_accentizer.load(omograph_model_size='turbo3', use_dictionary=True, tiny_mode=False)
    return global_phonemizer
    
def postprocess_phonemes(text):
    # 1. Исправляем мягкий знак (Ь → [j])
    text = re.sub(r"([бвгджзклмнпрстфхцчшщ])ʲ", r"\1j", text)  # Для согласных перед ʲ
    text = re.sub(r"([й])ʲ", r"\1", text)  # Для йотированных букв (йʲ → й)
    
    # 2. Удаляем артефакты (кавычки, неразрывные пробелы)
    text = re.sub(r"[\"«»“”„]", "", text)  # Удаляем кавычки
    text = re.sub(r"\s+", " ", text)       # Нормализуем пробелы
    
    # 3. Корректируем ударения (перемещаем ' за гласную)
    text = re.sub(r"([аеёиоуыэюя])([ˈ])", r"\2\1", text)  # 'е → е'
    
    # 4. Удаляем запрещенные символы (оставляем только IPA + ударения)
    allowed = r"a-zɐɑæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔˈˌːˑj"
    text = re.sub(f"[^{allowed}\s]", "", text)
    
    # 5. Убираем изолированные j (например, "j " → "")
    text = re.sub(r"\bj\b", "", text)  # Удаляем j как отдельный токен
    
    # 6. Проверяем пустые токены
    tokens = [t for t in text.split() if t]
    
    return " ".join(tokens)
    
def inference(text, ref_s, model, sampler, textclenaer, to_mel, device, model_params, global_phonemizer,global_accentizer, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):

    text = text.strip()
    accented_text = global_accentizer.process_all(text)
    ps = global_phonemizer.phonemize(accented_text)  # <<< УБРАЛИ [text] -> text #ps = global_phonemizer.phonemize([text]) 
    #ps = ' '.join(word_tokenize(ps[0])) 
    #ps = ' '.join(word_tokenize(ps)) 
    #ps = postprocess_phonemes(ps)
    tokens = textclenaer(ps)

    print(f"\nword_tokenize: '{text}' -> {ps}") 
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,
                         num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]

def ru_cleaner(text):
    if text is None:
        return []
    text = re.sub(r"[^a-zɐɑæəʙɓβɔɕçɗðɖɘɛɜɝɞɟʄɡɠɢʛɦɧħɥɨɪɬɫɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʈʧʉʊʋʌʍʎʏʐʑʒʓʔʕʖʡʢʘʙʚʛʜʝʞʟ]", "", text)
    return text.split()


def get_voice_dir(root="voices"):
    target = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../{root}')
    if not os.path.exists(target):
        target = os.path.dirname(f'./{root}/')

    os.makedirs(target, exist_ok=True)

    return target

def get_voice( name, dir=get_voice_dir(), load_latents=True, extensions=["wav", "mp3", "flac"] ):
	subj = f'{dir}/{name}/'
	if not os.path.isdir(subj):
		return
	files = os.listdir(subj)
	
	if load_latents:
		extensions.append("pth")

	voice = []
	for file in files:
		ext = os.path.splitext(file)[-1][1:]
		if ext not in extensions:
			continue

		voice.append(f'{subj}/{file}') 

	return sorted( voice )

def get_voice_list(dir=get_voice_dir(), append_defaults=False, extensions=["wav", "mp3", "flac", "pth", "opus", "m4a", "webm", "mp4"]):
	defaults = [ ]
	os.makedirs(dir, exist_ok=True)
	#res = sorted([d for d in os.listdir(dir) if d not in defaults and os.path.isdir(os.path.join(dir, d)) and len(os.listdir(os.path.join(dir, d))) > 0 ])

	res = []
	for name in os.listdir(dir):
		if name in defaults:
			continue
		if not os.path.isdir(f'{dir}/{name}'):
			continue
		if len(os.listdir(os.path.join(dir, name))) == 0:
			continue
		files = get_voice( name, dir=dir, extensions=extensions )

		if len(files) > 0:
			res.append(name)
		else:
			for subdir in os.listdir(f'{dir}/{name}'):
				if not os.path.isdir(f'{dir}/{name}/{subdir}'):
					continue
				files = get_voice( f'{name}/{subdir}', dir=dir, extensions=extensions )
				if len(files) == 0:
					continue
				res.append(f'{name}/{subdir}')

	res = sorted(res)
	
	if append_defaults:
		res = res + defaults
	
	return res

def load_models_webui(sigma_value, device="cpu", configuration_path="model_paths.yml"):
    lib_path = os.path.dirname(os.path.abspath(__file__))
    configuration_path = os.path.join(lib_path, configuration_path)
    config = load_configurations(configuration_path)
    ASR_config = os.path.join(lib_path, config.get('ASR_config', False))
    ASR_path = os.path.join(lib_path, config.get('ASR_path', False))
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    F0_path = os.path.join(lib_path, config.get('F0_path', False))
    pitch_extractor = load_F0_models(F0_path)

    BERT_path = os.path.join(lib_path, config.get('PLBERT_dir', False))
    from styletts2.Utils.PLBERT.util import load_plbert
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model_params.diffusion.dist.sigma_data = sigma_value
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    
    return model, model_params

# borrowed from tortoise, PR'ed by Jon

import re

def split_and_recombine_text(text, desired_length=200, max_length=300):
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[“”]', '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in '!?.\n ' and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in '!?\n' or (c == '.' and peek(1) in '\n ')):
            # seek forward if we have consecutive boundary markers but still within the max length
            while pos < len(text) - 1 and len(current) < max_length and peek(1) in '!?.':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?]*$', s)]

    return rv