
from etflow.data.datamodule import BaseDataModule


if __name__ == "__main__":
    # 指向我们之前跑脚本生成的顶层数据目录
    DATA_ROOT = "/remote-home/jinxianwang/tinghaoxia/RNA/Data/RNA"

    print("初始化 BaseDataModule...")
    dm = BaseDataModule(data_dir=DATA_ROOT, dataloader_args={"batch_size": 2, "num_workers": 0})
    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()

    for batch in train_loader:
        print("\n🎉 成功获取一个 Batch!")
        print("-" * 40)
        print(batch)
        print(f"Batch 中的分子数 (batch_size): {batch.num_graphs}")
        print(f"合并后的总原子数: {batch.num_nodes}")
        print(f"真实坐标形状 (pos): {batch.pos.shape}")
        print(f"预测坐标形状 (pos_pred): {batch.pos_pred.shape}")
        print(f"边索引形状 (edge_index): {batch.edge_index.shape}")

        # 为了确认你的数据确实进来了
        pdb_ids = batch.pdb_id
        print(f"当前 Batch 包含的 PDB IDs: {pdb_ids}")
        break  # 只测试第一个 Batch 即可