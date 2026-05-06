from utils import read_yaml
from etflow.data.datamodule import BaseDataModule
# 注意num_workers
if __name__ == "__main__":
    config = read_yaml("config/RNA_test.yaml")
    datamodule = BaseDataModule(**config["datamodule_args"])


    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()

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
    print(datamodule)



