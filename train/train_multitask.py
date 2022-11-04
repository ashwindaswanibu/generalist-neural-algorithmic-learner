
import clrs
import os
import shutil
import time
import numpy as np
import dataclasses
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import requests
import tensorflow as tf
import functools
from absl import app
from absl import flags
from absl import logging
from clrs._src import specs
from typing import Iterator
from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
from algorithms import algos

def train_multitask():
    for i in range(1000):
        for algo in algos:
            train_single_task(algo)


def train_single_task(algo):
  # Use canonical CLRS-30 samplers.
  clrs30_spec = clrs.CLRS30
#   logging.info('Using CLRS30 spec: %s', clrs30_spec)
  dataset_folder = '/tmp/CLRS30/CLRS30_v1.0.0'


  encode_hints = True
  decode_hints = True
  decode_diffs = True
  
  common_args = dict(folder=dataset_folder,
                     algorithm=algo,
                     batch_size=32)
  
  # Make full dataset pipeline run on CPU (including prefetching).
  with tf.device('/cpu:0'):

    train_sampler, _, spec = clrs.create_dataset(**common_args, split='train')
    train_sampler = train_sampler.as_numpy_iterator()
    train_sampler_for_eval = None

    val_sampler, val_samples, _ = clrs.create_dataset(
        **common_args, split='val')
    val_sampler = val_sampler.as_numpy_iterator()
    test_sampler, test_samples, _ = clrs.create_dataset(
        **common_args, split='test')
    test_sampler = test_sampler.as_numpy_iterator()

  processor_factory = clrs.get_processor_factory('mpnn',
                                                 use_ln=True,
                                                 nb_heads=1)
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=128,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      decode_diffs=decode_diffs,
      use_lstm=False,
      learning_rate=0.003,
      checkpoint_path='/tmp/CLRS30',
      freeze_processor=False,
      dropout_prob=0.0,
      hint_teacher_forcing_noise=0.0,
      )
  
  
  eval_model = clrs.models.BaselineModel(
        spec=spec,
        dummy_trajectory=next(val_sampler),
        **model_params
  )
    
  train_model = eval_model

  # Training loop.
  
  best_score = -1.0  # Ensure that there is overwriting
  rng_key = jax.random.PRNGKey(42)
  current_train_items = 0
  step = 0
  next_eval = 0

  while current_train_items < 10000:
    feedback = next(train_sampler)
    print(current_train_items)
    # Initialize model.
    if current_train_items == 0:
      
      t = time.time()
      train_model.init(feedback.features, 42 + 1)

    # Training step step.
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_loss = train_model.feedback(rng_key, feedback)
    rng_key = new_rng_key
    if current_train_items == 0:
      print("here")
      print('Compiled feedback step in %f s.', time.time() - t)
    
    examples_in_chunk = len(feedback.features.lengths)
    current_train_items += examples_in_chunk

    # Periodically evaluate model.
    if current_train_items >= next_eval:
      common_extras = {'examples_seen': current_train_items,
                       'step': step}
      eval_model.params = train_model.params
      # Training info.
      train_feedback = feedback
      rng_key, new_rng_key = jax.random.split(rng_key)
      train_stats = evaluate(
          rng_key,
          eval_model,
          train_feedback,
          spec=spec,
          extras=dict(loss=cur_loss, **common_extras),
          verbose=False,
      )
      rng_key = new_rng_key
      print('(train) step %d: %s', step, train_stats)

      # Validation info.
      rng_key, new_rng_key = jax.random.split(rng_key)
      val_stats = collect_and_eval(
          val_sampler,
          eval_model.predict,
          val_samples,
          rng_key,
          spec=spec,
          extras=common_extras)
      rng_key = new_rng_key
      print('(val) step %d: %s', step, val_stats)

      # If best scores, update checkpoint.
      score = val_stats['score']
      if score > best_score:
        logging.info('Saving new checkpoint...')
        best_score = score
        train_model.save_model('/content/drive/MyDrive/clrs/' + algo +'.pkl')
      next_eval += 320

    step += 1

  # Training complete, evaluate on test set.
  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model( '/content/drive/MyDrive/clrs/' +algo + '.pkl', only_load_processor=False)

  rng_key, new_rng_key = jax.random.split(rng_key)
  test_stats = collect_and_eval(
      test_sampler,
      eval_model.predict,
      test_samples,
      rng_key,
      spec=spec,
      extras=common_extras)
  rng_key = new_rng_key
  print('(test) step %d: %s', step, test_stats)

def collect_and_eval(sampler, predict_fn, sample_count, rng_key, spec, extras):
  """Collect batches of output and hint preds and evaluate them."""
  verbose = False
  processed_samples = 0
  preds = []
  hint_preds = []
  outputs = []
  hints = []
  lengths = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    outputs.append(feedback.outputs)
    rng_key, new_rng_key = jax.random.split(rng_key)
    cur_preds, (cur_hint_preds, _, _) = predict_fn(rng_key, feedback.features)
    preds.append(cur_preds)
    if verbose:
      hints.append(feedback.features.hints)
      lengths.append(feedback.features.lengths)
      hint_preds.append(cur_hint_preds)
    rng_key = new_rng_key
    processed_samples += 32
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  if verbose:
    # for hints, axis=1 because hints have time dimension first
    hints = _concat(hints, axis=1)
    lengths = _concat(lengths, axis=0)
    # for hint_preds, axis=0 because the time dim is unrolled as a list
    hint_preds = _concat(hint_preds, axis=0)

  return evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec,
                        extras)

def evaluate(rng_key, model, feedback, spec, extras=None, verbose=False):
  """Evaluates a model on feedback."""
  out = {}
  predictions, aux = model.predict(rng_key, feedback.features)
  out.update(clrs.evaluate(feedback.outputs, predictions))
  if model.decode_hints and verbose:
    hint_preds = [clrs.decoders.postprocess(spec, x) for x in aux[0]]
    out.update(clrs.evaluate_hints(feedback.features.hints,feedback.features.lengths, hint_preds))
  if extras:
    out.update(extras)
  if verbose:
    out.update(model.verbose_loss(feedback, aux))
  return {k: unpack(v) for k, v in out.items()}

def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v

def evaluate_preds(preds, outputs, hints, lengths, hint_preds, spec, extras):
  """Evaluates predictions against feedback."""
  out = {}
  out.update(clrs.evaluate(outputs, preds))
  if hint_preds:
    hint_preds = [clrs.decoders.postprocess(spec, x) for x in hint_preds]
    out.update(clrs.evaluate_hints(hints, lengths, hint_preds))
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis), *dps)    