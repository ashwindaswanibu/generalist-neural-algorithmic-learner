
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


def process_pred_as_input(spec, sample_iterator):
  """Move pred_h hint to pred input."""
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      features = feedback.features
      pred_h = [h for h in features.hints if h.name == 'pred_h']
      if pred_h:
        assert len(pred_h) == 1
        pred_h = pred_h[0]
        hints = [h for h in features.hints if h.name != 'pred_h']
        for i in range(len(features.lengths)):
          assert np.sum(np.abs(pred_h.data[1:int(features.lengths[i]), i] -
                               pred_h.data[0, i])) == 0.0
        inputs = tuple(features.inputs) + (
            probing.DataPoint(name='pred', location=pred_h.location,
                              type_=pred_h.type_, data=pred_h.data[0]),)
        features = features._replace(inputs=tuple(inputs),
                                     hints=tuple(hints))
        feedback = feedback._replace(features=features)
      yield feedback

  new_spec = {}
  for k in spec:
    if k == 'pred_h':
      assert spec[k] == (specs.Stage.HINT, specs.Location.NODE,
                         specs.Type.POINTER)
      new_spec['pred'] = (specs.Stage.INPUT, specs.Location.NODE,
                          specs.Type.POINTER)
    else:
      new_spec[k] = spec[k]

  return new_spec, _iterate()

def process_random_pos(sample_iterator, rng):
  """Randomize the `pos` input from a sampler.
  The `pos` input is, by default, a scalar uniformly spaced between 0 and 1
  across the nodes. The exception are string algorithms (naive_string_matcher,
  kmp_string_matcher and lcs_length), where the `pos` sequence is split into
  needle and haystack (or first and second string, for lcs_length). To avoid 
  overfitting to these linearly spaced values during training, we replaced them 
  with random values, uniformly sampled in [0, 1].
  Parameters:
    sample_iterator: An iterator producing samples with non-random `pos` inputs.
    rng: Numpy random generator
  Returns:
    An iterator returning the samples with randomized `pos` inputs.
  """
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      inputs = feedback.features.inputs
      pos, = [x for x in inputs if x.name == 'pos']
      batch_size, num_nodes = pos.data.shape
      unsorted = rng.uniform(size=(batch_size, num_nodes))
      new_pos = []
      for i in range(batch_size):  # we check one example at a time.
        # We find if there are splits in the pos sequence, marked by zeros.
        # We know there will always be at least 1 zero, if there's no split.
        split, = np.where(pos.data[i] == 0)
        split = np.concatenate([split, [num_nodes]])
        # We construct the randomized pos by sorting the random values in each
        # split and concatenating them.
        new_pos.append(
            np.concatenate([np.sort(unsorted[i, split[j]:split[j+1]])
                            for j in range(len(split) - 1)]))
      pos.data = np.array(new_pos)
      inputs = [(pos if x.name == 'pos' else x) for x in inputs]
      features = feedback.features._replace(inputs=inputs)
      feedback = feedback._replace(features=features)
      yield feedback

  return _iterate()

def _preprocess_permutations(probes, enforce_permutations):
  """Replace should-be permutations with proper permutation pointer + mask."""
  output = []
  for x in probes:
    if x.type_ != specs.Type.SHOULD_BE_PERMUTATION:
      output.append(x)
      continue
    assert x.location == specs.Location.NODE
    if enforce_permutations:
      new_x, mask = probing.predecessor_to_cyclic_predecessor_and_first(x.data)
      output.append(
          probing.DataPoint(
              name=x.name,
              location=x.location,
              type_=specs.Type.PERMUTATION_POINTER,
              data=new_x))
      output.append(
          probing.DataPoint(
              name=x.name + '_mask',
              location=x.location,
              type_=specs.Type.MASK_ONE,
              data=mask))
    else:
      output.append(probing.DataPoint(name=x.name, location=x.location,
                                      type_=specs.Type.POINTER, data=x.data))
  return output


def process_permutations(spec, sample_iterator, enforce_permutations):
  """Replace should-be permutations with proper permutation pointer + mask."""
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      features = feedback.features
      inputs = _preprocess_permutations(features.inputs, enforce_permutations)
      hints = _preprocess_permutations(features.hints, enforce_permutations)
      outputs = _preprocess_permutations(feedback.outputs, enforce_permutations)
      features = features._replace(inputs=tuple(inputs),
                                   hints=tuple(hints))
      feedback = feedback._replace(features=features,
                                   outputs=outputs)
      yield feedback

  new_spec = {}
  for k in spec:
    if (spec[k][1] == specs.Location.NODE and
        spec[k][2] == specs.Type.SHOULD_BE_PERMUTATION):
      if enforce_permutations:
        new_spec[k] = (spec[k][0], spec[k][1], specs.Type.PERMUTATION_POINTER)
        new_spec[k + '_mask'] = (spec[k][0], spec[k][1], specs.Type.MASK_ONE)
      else:
        new_spec[k] = (spec[k][0], spec[k][1], specs.Type.POINTER)
    else:
      new_spec[k] = spec[k]

  return new_spec, _iterate()
