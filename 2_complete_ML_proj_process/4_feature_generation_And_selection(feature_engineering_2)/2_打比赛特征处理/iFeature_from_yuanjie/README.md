## 特征工程工具包

### Install
```bash
pip install git+https://github.com/Jie-Yuan/iFeature.git
```

```bash
ssh-keygen -t rsa -C "313303303@qq.com" #一路回车

head ~/.ssh/id_rsa.pub # 复制添加至Github SSH keys

git config --global color.ui true
git config --global user.name "Jie-Yuan"
git config --global user.email "313303303@qq.com"
git config --list
```

```
_clf = LGBMClassifier(n_estimators=1)
X = iris.data[:100, :]
y = iris.target[:100]
_clf.fit(X, y)
show_info = ['split_gain', 'internal_value', 'internal_count', 'leaf_count']
lgb.plot_tree(_clf.booster_, figsize=(60, 80), show_info=show_info)

model = _clf.booster_.dump_model()
tree_infos = model['tree_info'] # xgb_._Booster.get_dump()
```
