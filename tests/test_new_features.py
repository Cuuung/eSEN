"""
快速测试两个新功能：
1. 滚动保存 checkpoint（keep_last_k_checkpoints）
2. NaN 梯度自动中断训练
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch


# ─────────────────────────────────────────────────────────────────────────────
# 测试 1：滚动保存
# ─────────────────────────────────────────────────────────────────────────────
class TestRollingCheckpoint(unittest.TestCase):
    def _make_trainer(self, tmpdir, keep_k):
        """构造一个最简 MLIPTrainer stub，只测试 _save_rolling_checkpoint。"""
        from fairchem.core.models.esen.trainers.trainer import MLIPTrainer

        trainer = object.__new__(MLIPTrainer)
        trainer.config = {
            "optim": {"keep_last_k_checkpoints": keep_k},
            "cmd": {"checkpoint_dir": tmpdir},
        }
        trainer.step = 0

        # save() 只创建空文件，模拟真实行为
        def fake_save(checkpoint_file, training_state):
            path = os.path.join(tmpdir, checkpoint_file)
            open(path, "w").close()

        trainer.save = fake_save
        return trainer

    def test_rolling_keeps_last_k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            keep_k = 3
            trainer = self._make_trainer(tmpdir, keep_k)

            # 模拟 5 个 epoch 保存
            for epoch_int in range(5):
                trainer.step = (epoch_int + 1) * 100
                trainer._save_rolling_checkpoint(epoch_int)

            files = sorted(os.listdir(tmpdir))
            print(f"  保留的文件：{files}")
            self.assertEqual(len(files), keep_k,
                             f"期望保留 {keep_k} 个文件，实际 {len(files)} 个")

    def test_rolling_deletes_oldest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            keep_k = 2
            trainer = self._make_trainer(tmpdir, keep_k)

            for epoch_int in range(4):
                trainer.step = (epoch_int + 1) * 100
                trainer._save_rolling_checkpoint(epoch_int)

            files = sorted(os.listdir(tmpdir))
            print(f"  保留的文件：{files}")
            # 最新的两个 epoch 应该存在（epoch3, epoch4）
            self.assertTrue(any("epoch3" in f for f in files))
            self.assertTrue(any("epoch4" in f for f in files))
            # 最老的 epoch1 应该被删掉
            self.assertFalse(any("epoch1" in f for f in files))


# ─────────────────────────────────────────────────────────────────────────────
# 测试 2：NaN 梯度检测
# ─────────────────────────────────────────────────────────────────────────────
class TestNaNGradientDetection(unittest.TestCase):
    def _make_trainer(self):
        """构造一个最简 MLIPTrainer stub，只测试 _backward 的 NaN 分支。"""
        from fairchem.core.models.esen.trainers.trainer import MLIPTrainer

        trainer = object.__new__(MLIPTrainer)
        trainer.step = 1
        trainer.epoch = 0.01
        trainer.device = torch.device("cpu")
        trainer.clip_grad_norm = 10.0
        trainer.scaler = None
        trainer.ema = None
        trainer.logger = None
        trainer._nan_grad_detected = False

        # model.parameters() 返回一个假参数
        param = torch.nn.Parameter(torch.tensor([1.0]))
        param.grad = torch.tensor([1.0])
        mock_model = MagicMock()
        mock_model.parameters.return_value = [param]
        mock_model.shared_parameters = []
        trainer.model = mock_model

        trainer.optimizer = MagicMock()
        return trainer

    def test_nan_grad_sets_flag(self):
        trainer = self._make_trainer()

        # patch clip_grad_norm_ 返回 NaN，模拟梯度爆炸
        with patch("torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(float("nan"))):
            # distutils.all_reduce 在单卡下直接返回输入
            with patch("fairchem.core.common.distutils.all_reduce", side_effect=lambda x, **kw: x):
                loss = MagicMock()
                loss.backward = MagicMock()
                trainer._backward(loss)

        self.assertTrue(trainer._nan_grad_detected,
                        "_nan_grad_detected 应该被设为 True")
        trainer.optimizer.step.assert_not_called()
        print("  NaN 检测到，optimizer.step 未被调用 ✓")

    def test_normal_grad_no_flag(self):
        trainer = self._make_trainer()

        # patch clip_grad_norm_ 返回正常值
        with patch("torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(0.5)):
            with patch("fairchem.core.common.distutils.all_reduce", side_effect=lambda x, **kw: x):
                loss = MagicMock()
                loss.backward = MagicMock()
                trainer._backward(loss)

        self.assertFalse(trainer._nan_grad_detected,
                         "正常梯度不应触发 _nan_grad_detected")
        trainer.optimizer.step.assert_called_once()
        print("  正常梯度，optimizer.step 被调用一次 ✓")


if __name__ == "__main__":
    unittest.main(verbosity=2)
