# Changelog

<!--next-version-placeholder-->

## v1.2.0 (2023-03-21)
### Feature
* Add presets ([#55](https://github.com/34j/so-vits-svc-fork/issues/55)) ([`e8adcc6`](https://github.com/34j/so-vits-svc-fork/commit/e8adcc621f6caf5f4b20846575b3559c032ed47f))

## v1.1.1 (2023-03-21)
### Fix
* **deps:** Update dependency gradio to v3.23.0 ([#54](https://github.com/34j/so-vits-svc-fork/issues/54)) ([`a2bdb48`](https://github.com/34j/so-vits-svc-fork/commit/a2bdb48b436d206b30bb72409852c0b30d6811e9))

## v1.1.0 (2023-03-21)
### Feature
* Enhance RealtimeVC ([#52](https://github.com/34j/so-vits-svc-fork/issues/52)) ([`81551ce`](https://github.com/34j/so-vits-svc-fork/commit/81551ce9c6fb7924d184c3c5a4cf9035168b28d2))

### Documentation
* Update gui screenshot ([#53](https://github.com/34j/so-vits-svc-fork/issues/53)) ([`58d06aa`](https://github.com/34j/so-vits-svc-fork/commit/58d06aa7460dd75ef793da295bf7651ae9940814))

## v1.0.2 (2023-03-21)
### Fix
* **deps:** Update dependency scipy to v1.10.1 ([#35](https://github.com/34j/so-vits-svc-fork/issues/35)) ([`e0253bf`](https://github.com/34j/so-vits-svc-fork/commit/e0253bf1e655f86be605395a18f343763d975101))

## v1.0.1 (2023-03-20)
### Documentation
* Add ThrowawayAccount01 as a contributor for bug ([#47](https://github.com/34j/so-vits-svc-fork/issues/47)) ([`15e31fa`](https://github.com/34j/so-vits-svc-fork/commit/15e31fa806249d45235918fa62a48a86c43538cb))
* Add BlueAmulet as a contributor for ideas ([#46](https://github.com/34j/so-vits-svc-fork/issues/46)) ([`a3bcb2b`](https://github.com/34j/so-vits-svc-fork/commit/a3bcb2be2992c98bcc2485082c19009c74cb3194))

### Performance
* **inference_main:** Do dummy inference before running vc ([#45](https://github.com/34j/so-vits-svc-fork/issues/45)) ([`4066c43`](https://github.com/34j/so-vits-svc-fork/commit/4066c4334b107062d2daa7c9dc00600a56c6e553))

## v1.0.0 (2023-03-20)
### Fix
* Fix default dataset path ([#43](https://github.com/34j/so-vits-svc-fork/issues/43)) ([`ac47fed`](https://github.com/34j/so-vits-svc-fork/commit/ac47fede2581d375c2be9c28102961f19f5a9aa1))

### Breaking
* the behaviour of preprocess_resample changes when there is a folder ./dataset_raw/44k and "44k" is no longer allowed as a speaker name in some conditions ([`ac47fed`](https://github.com/34j/so-vits-svc-fork/commit/ac47fede2581d375c2be9c28102961f19f5a9aa1))

## v0.8.2 (2023-03-20)
### Fix
* Fix compute_f0_crepe returning wrong length ([#42](https://github.com/34j/so-vits-svc-fork/issues/42)) ([`afb42b0`](https://github.com/34j/so-vits-svc-fork/commit/afb42b019ccd133876a2c55cf01007950a733d8c))

## v0.8.1 (2023-03-20)
### Fix
* **deps:** Update dependency librosa to v0.10.0 ([#40](https://github.com/34j/so-vits-svc-fork/issues/40)) ([`8e92f71`](https://github.com/34j/so-vits-svc-fork/commit/8e92f71b2820628f0f8583e6bc455d8f753f4302))

## v0.8.0 (2023-03-20)
### Feature
* Add more f0 calculation methods ([#39](https://github.com/34j/so-vits-svc-fork/issues/39)) ([`6b3b20d`](https://github.com/34j/so-vits-svc-fork/commit/6b3b20dfd609d81cb1184b7c8e8865a58f8d45f9))

## v0.7.1 (2023-03-20)
### Fix
* **deps:** Update dependency gradio to v3.22.1 ([#33](https://github.com/34j/so-vits-svc-fork/issues/33)) ([`f09fc23`](https://github.com/34j/so-vits-svc-fork/commit/f09fc23ca82519cc095509d4d4760561424a17ec))

## v0.7.0 (2023-03-20)
### Feature
* **preprocessing:** Allow nested dataset ([#19](https://github.com/34j/so-vits-svc-fork/issues/19)) ([`0433151`](https://github.com/34j/so-vits-svc-fork/commit/0433151d94c4da8e84a0183bdd47f1e08ea3c462))

## v0.6.3 (2023-03-20)
### Fix
* **deps:** Update dependency torch to v1.13.1 ([#27](https://github.com/34j/so-vits-svc-fork/issues/27)) ([`8826d68`](https://github.com/34j/so-vits-svc-fork/commit/8826d6870e223e7969baa069bf12235e0deec0b7))
* **deps:** Update dependency torchaudio to v0.13.1 ([#28](https://github.com/34j/so-vits-svc-fork/issues/28)) ([`989f5d9`](https://github.com/34j/so-vits-svc-fork/commit/989f5d903b47ba9b0ea1d0fe37cbfe76edf0a811))

### Documentation
* **readme:** Update notes about VRAM caps ([#18](https://github.com/34j/so-vits-svc-fork/issues/18)) ([`0a245f4`](https://github.com/34j/so-vits-svc-fork/commit/0a245f4ee69bd0d4371836367becf0fe409431e2))

## v0.6.2 (2023-03-19)
### Fix
* Use hubert preprocess force_rebuild argument ([#15](https://github.com/34j/so-vits-svc-fork/issues/15)) ([`87cf807`](https://github.com/34j/so-vits-svc-fork/commit/87cf807496248e2c7b859069f81aa040e86aec59))

### Documentation
* Add GarrettConway as a contributor for bug ([#17](https://github.com/34j/so-vits-svc-fork/issues/17)) ([`31d9671`](https://github.com/34j/so-vits-svc-fork/commit/31d9671207143fd06b8db148802d1e27874151ce))
* **notebook:** Launch tensorboard ([#16](https://github.com/34j/so-vits-svc-fork/issues/16)) ([`52229ba`](https://github.com/34j/so-vits-svc-fork/commit/52229ba0fe9458e37b45287c0a716c7cd36adbd6))
* Add 34j as a contributor for example, infra, and 6 more ([#14](https://github.com/34j/so-vits-svc-fork/issues/14)) ([`1b90378`](https://github.com/34j/so-vits-svc-fork/commit/1b903783b4b89f2f5a4fc2e1b47f3eade0c0402f))
* Add GarrettConway as a contributor for code ([#13](https://github.com/34j/so-vits-svc-fork/issues/13)) ([`716813f`](https://github.com/34j/so-vits-svc-fork/commit/716813fbff85ab4609d8ec3f374b78c6551877e5))

## v0.6.1 (2023-03-19)
### Performance
* **preprocessing:** Better performance ([#12](https://github.com/34j/so-vits-svc-fork/issues/12)) ([`668c8e1`](https://github.com/34j/so-vits-svc-fork/commit/668c8e1f18cefb0ebd2fb2f1d6572ce4d37d1102))

## v0.6.0 (2023-03-18)
### Feature
* Configurable input and output devices ([#11](https://github.com/34j/so-vits-svc-fork/issues/11)) ([`a822a60`](https://github.com/34j/so-vits-svc-fork/commit/a822a6098d322ff37725eee19d17758f72a6db49))

### Documentation
* **notebook:** Fix notebook ([#9](https://github.com/34j/so-vits-svc-fork/issues/9)) ([`427b4c1`](https://github.com/34j/so-vits-svc-fork/commit/427b4c1c6e0482345b17fedb018f7a18db68ccc5))
* Update notebook ([#8](https://github.com/34j/so-vits-svc-fork/issues/8)) ([`ae3e471`](https://github.com/34j/so-vits-svc-fork/commit/ae3e4710aac41555f00ddcdfbcf5a5e925afb718))

## v0.5.0 (2023-03-18)
### Feature
* **gui:** Remember last directory (misc) ([`92558da`](https://github.com/34j/so-vits-svc-fork/commit/92558da2f0e4eb24a8de412fb7e22dc3530b648a))
* **__main__:** Show defaults ([`3d298df`](https://github.com/34j/so-vits-svc-fork/commit/3d298df91bdfca230959603da74331b5eef4d487))

### Fix
* **__main__:** Fix option names ([`7ff34fe`](https://github.com/34j/so-vits-svc-fork/commit/7ff34fe623dde6b0a684c45cf33dc54118f9a800))

### Documentation
* **readme:** Update README.md ([#6](https://github.com/34j/so-vits-svc-fork/issues/6)) ([`b988101`](https://github.com/34j/so-vits-svc-fork/commit/b98810194703b6bb0ede03a00c460eeecdab5131))

## v0.4.1 (2023-03-18)
### Fix
* Call init_logger() ([#5](https://github.com/34j/so-vits-svc-fork/issues/5)) ([`e6378f1`](https://github.com/34j/so-vits-svc-fork/commit/e6378f12e747e618ff90ece1552d09c0d0714d41))

## v0.4.0 (2023-03-18)
### Feature
* Enhance realtime algorythm ([#4](https://github.com/34j/so-vits-svc-fork/issues/4)) ([`d789a12`](https://github.com/34j/so-vits-svc-fork/commit/d789a12308784473ae5d09e0b73fa15bf7554de1))

## v0.3.0 (2023-03-17)
### Feature
* Add gui ([#3](https://github.com/34j/so-vits-svc-fork/issues/3)) ([`34aec2b`](https://github.com/34j/so-vits-svc-fork/commit/34aec2b98ee4ef82ef488129b61a7952af5226a3))

### Documentation
* Update notebook ([`7b74606`](https://github.com/34j/so-vits-svc-fork/commit/7b74606508cfb7e45224cbd76f3de9c43c8b4309))

## v0.2.1 (2023-03-17)
### Fix
* **notebook:** Fix notebook ([`3ed00cc`](https://github.com/34j/so-vits-svc-fork/commit/3ed00cc66d4f66e045f61fc14937cb9160eee556))

## v0.2.0 (2023-03-17)
### Feature
* Realtime inference ([#2](https://github.com/34j/so-vits-svc-fork/issues/2)) ([`4dea1ae`](https://github.com/34j/so-vits-svc-fork/commit/4dea1ae51fe2e47a3f41556bdbe3fefd033d729a))

## v0.1.0 (2023-03-17)
### Feature
* Main feat ([#1](https://github.com/34j/so-vits-svc-fork/issues/1)) ([`faa990c`](https://github.com/34j/so-vits-svc-fork/commit/faa990ce6411d8b4e8b3d2d48c4b532b76ff7800))
