"""CPU regression checks: PYTHONPATH=. python scripts/check_poultry_surface_v2.py."""

from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np
import torch

from mjlab.terrains import TerrainGenerator
from src.tasks.duet.config.g1_23dof.poultry_surface import (
  unitree_g1_23dof_duet_poultry_surface_env_cfg,
)
from src.tasks.duet.duet_env_cfg import DUET_TERRAINS_CFG, _ROUGH
from src.tasks.duet.mdp.poultry_surface import (
  PoultrySurfaceActionCfg, poultry_clearance, poultry_lift_target,
  poultry_swing_height,
)


class PoultrySurfaceChecks(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(42)
    self.cfg = unitree_g1_23dof_duet_poultry_surface_env_cfg()
    self.robot = SimpleNamespace(
      find_bodies=lambda *a, **kw: ([0, 1], []),
      write_external_wrench_to_sim=Mock(),
    )
    geoms = ''.join(
      f'<geom name="robot/{side}_foot{i}_collision" type="capsule" size=".01 .02"/>'
      for side in ('left', 'right') for i in range(1, 8)
    )
    model = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody>{geoms}</worldbody></mujoco>')
    self.env = SimpleNamespace(
      num_envs=128, device='cpu', cfg=self.cfg, scene={'robot': self.robot},
      sim=SimpleNamespace(mj_model=model),
    )
    self.surface = self.cfg.actions['poultry_surface'].build(self.env)

  def tile_points(self, row=4):
    tg = self.surface.terrain_cfg
    points = torch.zeros(self.env.num_envs, 2, tg.num_cols, 3)
    points[..., 0] = (row + .5 - tg.num_rows / 2) * tg.size[0]
    points[..., 1] = (torch.arange(tg.num_cols) + .5 - tg.num_cols / 2) * tg.size[1]
    return points

  def test_random_depth_and_reset(self):
    s = self.surface
    before = s.litter_depth.clone()
    self.assertTrue(((before >= 0) & (before <= .06)).all())
    self.assertEqual(before.unique().numel(), self.env.num_envs)
    ids = torch.tensor([1, 4, 9, 20])
    s.reset(ids)
    self.assertTrue((s.litter_depth[ids] != before[ids]).all())
    before[ids] = s.litter_depth[ids]
    torch.testing.assert_close(s.litter_depth, before, rtol=0, atol=0)
    s.reset()
    self.assertTrue((s.litter_depth != before).all())
    self.assertTrue(((s.litter_depth >= 0) & (s.litter_depth <= .06)).all())
    with torch.inference_mode():
      s.reset(slice(0, 2))
    s.reset(torch.tensor([], dtype=torch.long))
    s.reset()
    print('PASS: random initialization, subset/all/slice/empty resets; range [0, 0.06]')

  def test_column_masks_and_row_independence(self):
    s = self.surface
    for row in (0, 4, 9):
      valid, depth = s.terrain_at(self.tile_points(row))
      expected = s.litter_depth[:, None, None] * s.soft_columns[None, None, :]
      torch.testing.assert_close(depth, expected.expand_as(depth), rtol=0, atol=0)
      self.assertTrue((valid == s.flat_columns).all())
    outside = self.tile_points()
    outside[..., 0] = 1e6
    valid, depth = s.terrain_at(outside)
    self.assertFalse(valid.any())
    self.assertFalse(depth.any())
    print('PASS: per-env soft depth broadcasts across feet/points, independent of row; flat/HF/border zero')

  def test_row_legacy_and_disabled(self):
    s = self.surface
    s.cfg.litter_depth_mode = 'row'
    s.cfg.max_layer_height = .12
    tg = s.terrain_cfg
    for row in range(tg.num_rows):
      p = self.tile_points(row)
      _, actual = s.terrain_at(p)
      r = torch.floor(p[..., 0] / tg.size[0] + tg.num_rows / 2).long()
      expected = .12 * r.clamp(0, tg.num_rows - 1) / max(tg.num_rows - 1, 1)
      expected *= s.soft_columns
      torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for mode in ('row', 'random'):
      s.cfg.litter_depth_mode = mode
      s.cfg.litter_enabled = False
      self.assertFalse(s.terrain_at(self.tile_points())[1].any())
    s.terrain_cfg = None
    valid, depth = s.terrain_at(self.tile_points_from_shape())
    self.assertTrue(valid.all())
    self.assertFalse(depth.any())
    print('PASS: exact legacy row formula at all rows, both disabled modes and plane terrain zero')

  def tile_points_from_shape(self):
    return torch.zeros(self.env.num_envs, 2, 14, 3)

  def test_pinned_depth_and_play(self):
    self.assertEqual(PoultrySurfaceActionCfg(entity_name='robot').litter_depth_mode, 'row')
    play = unitree_g1_23dof_duet_poultry_surface_env_cfg(play=True)
    self.assertEqual(play.actions['poultry_surface'].litter_depth_mode, 'row')
    for play in (False, True):
      cfg = unitree_g1_23dof_duet_poultry_surface_env_cfg(
        play=play, litter_depth_mode='random', litter_depth_range=(.045, .045),
      )
      s = cfg.actions['poultry_surface'].build(self.env)
      s.reset()
      torch.testing.assert_close(s.litter_depth, torch.full_like(s.litter_depth, .045))
    print('PASS: play keeps row mode; training and play can pin (0.045, 0.045)')

  def test_invalid_depth_config(self):
    cfg = self.surface.cfg
    for kw in ({'litter_depth_mode': 'invalid'}, {'litter_depth_range': (-.01, .06)},
               {'litter_depth_range': (.06, .01)}, {'litter_depth_range': (0, float('nan'))}):
      with self.assertRaises(ValueError):
        replace(cfg, **kw).build(self.env)
    print('PASS: invalid depth mode, negative/reversed/NaN ranges rejected')

  def test_targets_and_both_rewards(self):
    layer = torch.tensor([0, .02, .04, .05, .06, .12])[:, None, None].expand(-1, 2, 14)
    expected = torch.tensor([.09, .09, .09, .10, .11, .11])[:, None].expand(-1, 2)
    target = poultry_lift_target(layer, above_litter=.05)
    torch.testing.assert_close(target, expected)
    self.assertEqual(poultry_lift_target(layer), .09)
    self.assertEqual(poultry_lift_target(layer, target_height=.07), .07)
    sample = dict(layer=layer, lowest=expected, velocity=torch.ones(6, 2, 14, 3),
                  active=torch.ones(6, 2, dtype=torch.bool), landing=torch.ones(6, 2, dtype=torch.bool))
    state = SimpleNamespace(sample=lambda: sample, last_apex=expected)
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: state), step_dt=.02)
    torch.testing.assert_close(poultry_clearance(env, above_litter=.05), torch.zeros(6), atol=1e-6, rtol=0)
    torch.testing.assert_close(poultry_swing_height(env, above_litter=.05), torch.zeros(6), atol=1e-6, rtol=0)
    sample['lowest'] = expected - .02
    cost = poultry_clearance(env, above_litter=.05)
    sample['velocity'] *= 2
    torch.testing.assert_close(poultry_clearance(env, above_litter=.05), cost * 2)
    legacy = ((sample['lowest'] - .09).abs() * sample['velocity'][..., :2].norm(dim=-1).mean(-1)).sum(-1)
    torch.testing.assert_close(poultry_clearance(env), legacy, atol=0, rtol=0)
    legacy_apex = ((expected / .09 - 1).square().clamp_max(4)).sum(-1) / .02
    torch.testing.assert_close(poultry_swing_height(env), legacy_apex, atol=0, rtol=0)
    print('PASS: depths [0,2,4,5,6,12] cm -> targets [9,9,9,10,11,11] cm; both rewards, legacy constants, speed weighting')

  def test_actual_generator_layout_and_isolation(self):
    before = asdict(DUET_TERRAINS_CFG)
    rough_before = {k: asdict(v) for k, v in _ROUGH.items()}
    cfg = unitree_g1_23dof_duet_poultry_surface_env_cfg()
    self.assertEqual(before, asdict(DUET_TERRAINS_CFG))
    self.assertEqual(rough_before, {k: asdict(v) for k, v in _ROUGH.items()})
    tg = cfg.scene.terrain.terrain_generator
    for name, sub in tg.sub_terrains.items():
      self.assertIsNot(sub, DUET_TERRAINS_CFG.sub_terrains[name])
      self.assertEqual(sub.flat_patch_sampling, DUET_TERRAINS_CFG.sub_terrains[name].flat_patch_sampling)
    # Execute mjlab's real curriculum assignment, replacing only expensive geometry.
    gen = object.__new__(TerrainGenerator)
    gen.cfg = tg
    gen.np_rng = np.random.default_rng(0)
    gen.terrain_origins = np.zeros((tg.num_rows, tg.num_cols, 3))
    names = {id(s): name for name, s in tg.sub_terrains.items()}
    mapping = {}
    def record(spec, position, difficulty, sub, row, col):
      mapping[col] = names[id(sub)]
      return np.zeros(3)
    gen._create_terrain_geom = record
    gen._generate_curriculum_terrains(None)
    self.assertEqual(Counter(mapping.values()), dict(flat=3, soft_floor=12, hf_pyramid_slope=1,
      hf_pyramid_slope_inv=1, random_rough=1, wave_terrain=2))
    for col, name in mapping.items():
      self.assertEqual(bool(self.surface.flat_columns[col]), name in ('flat', 'soft_floor'))
      self.assertEqual(bool(self.surface.soft_columns[col]), name == 'soft_floor')
    print('COLUMN MAPPING:', mapping)
    print('FLAT MASK (reward-valid, includes soft):', self.surface.flat_columns.int().tolist())
    print('SOFT MASK:', self.surface.soft_columns.int().tolist())
    print('PASS: actual mjlab column assignment matches masks; shared configs and spawn patches preserved')


if __name__ == '__main__':
  unittest.main(verbosity=2)
