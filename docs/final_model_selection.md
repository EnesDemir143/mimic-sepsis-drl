# Final CQL Model Selection

## Planla uyumlu kısa açıklama

Bu seçim `docs/final_project_plan_proposal.md` dosyasındaki iki aşamalı protokole göre yapılmıştır:

- Stage 1: 24 konfigürasyon tek seed ile validation üzerinde taranır.
- Stage 2: Stage 1'den gelen 6 aday validation üzerinde doğrulanır.
- Final model/config/checkpoint seçimi yalnızca validation sonuçlarına göre yapılır.
- Held-out test set model seçiminde, checkpoint seçiminde veya debugging'de kullanılmaz.
- Test set sadece final model seçildikten sonra tek final evaluation komutu ile kullanılmalıdır.

Mevcut `scripts/evaluate_cql_sweep.py --stage final` doğrudan final için kullanılmamalıdır; çünkü manifest içindeki tüm run/checkpoint adaylarını test split üzerinde dolaşır ve test FQE ile checkpoint seçer. Bu, final proposal'daki "test once after validation selection" kuralına aykırıdır. Bu yüzden tek seçilmiş checkpoint'i değerlendiren ayrı runner eklendi: `scripts/evaluate_final_selected_policy.py`.

## Stage 1 özeti

Kaynak: `runs/cql_sweep/stage1_evaluation.json`

Stage 1, 24 CQL konfigürasyonunu seed=42 ile validation split üzerinde taramıştır. En iyi checkpoint her run için validation FQE'ye göre seçilmiştir. Stage 1 sonuçları sadece aday belirleme amacıyla kullanılmıştır.

Top 6 Stage 1 adayı:

| Rank | reward_variant | learning_rate | cql_alpha | seed | best_epoch | validation FQE | WIS | ESS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | sparse | 1e-3 | 0.1 | 42 | 200 | 18.539986 | 12.443313 | 5.376691 |
| 2 | sparse | 1e-4 | 0.05 | 42 | 200 | 18.277767 | -13.174457 | 1.001159 |
| 3 | shaped | 1e-4 | 1.0 | 42 | 200 | 18.258302 | -13.162246 | 1.001805 |
| 4 | shaped | 1e-3 | 0.1 | 42 | 200 | 18.209286 | 12.497989 | 5.363235 |
| 5 | shaped | 1e-3 | 1.0 | 42 | 200 | 18.098364 | 12.481094 | 6.667413 |
| 6 | sparse | 3e-4 | 1.0 | 42 | 200 | 17.910558 | 6.767402 | 4.869092 |

## Stage 2 özeti

Kaynaklar:

- `runs/cql_sweep/stage2_manifest.json`
- `runs/cql_sweep/stage2_evaluation.json`

Stage 2 manifest, Stage 1'de seçilen 6 konfigürasyonu ek seed'lerle eğitmiştir: 123, 456, 789, 1024. Mevcut Stage 2 evaluation artifact'i konfigürasyon başına aggregate multi-seed tablo yerine validation FQE'ye göre en iyi seed/checkpoint entry'lerini raporlamaktadır. Dosyada CI, support/coverage, low-support action rate veya config-level seed standard deviation alanları yoktur. Bu nedenle seçimde mevcut Stage 2 validation FQE, WIS ve ESS değerleri kullanılabilir; fakat raporda stability/CI/support bilgisi "mevcut artifact'te yok" olarak belirtilmelidir.

Stage 2 validation sıralaması:

| Rank | reward_variant | learning_rate | cql_alpha | seed | best_epoch | validation FQE | WIS | ESS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | sparse | 1e-4 | 0.05 | 1024 | 200 | 15.670790 | 5.379920 | 7.824754 |
| 2 | shaped | 1e-3 | 0.1 | 1024 | 200 | 15.661057 | 6.026481 | 8.182755 |
| 3 | shaped | 1e-3 | 1.0 | 1024 | 200 | 15.633579 | 1.626519 | 5.172369 |
| 4 | shaped | 1e-4 | 1.0 | 1024 | 200 | 15.622208 | 6.805624 | 9.243024 |
| 5 | sparse | 1e-3 | 0.1 | 1024 | 200 | 15.582195 | 5.927978 | 8.424084 |
| 6 | sparse | 3e-4 | 1.0 | 1024 | 200 | 15.320593 | 8.831675 | 7.135866 |

## Final model seçim kriterleri

Seçim sadece Stage 2 validation sonuçlarına göre yapılmıştır. Test set kullanılmamıştır.

Kriterler:

1. Primary metric: validation FQE under common terminal survival/death utility.
2. WIS ve ESS diagnostic olarak değerlendirildi; FQE'yi tek başına kör biçimde seçmemek için özellikle yakın adaylarda WIS/ESS kontrol edildi.
3. Best checkpoint epoch validation FQE'ye göre alınmıştır; final epoch varsayımı yapılmamıştır. Mevcut sonuçlarda seçilen adayın best_epoch değeri 200'dür.
4. Seed/stability: Mevcut Stage 2 output config-level seed mean/std veya CI içermediği için stability doğrudan nicel olarak raporlanamamaktadır. Ancak seçilen checkpoint Stage 2 doğrulama setinde seed=1024 için en yüksek validation FQE'yi vermiştir.
5. Support/coverage: Mevcut Stage 2 output behavior-support mass veya low-support action rate alanı içermemektedir. ESS diagnostic'i support açısından mevcut en yakın sinyaldir.

## Seçilen final model

Seçilen final checkpoint:

`checkpoints/cql_sweep/cql_s1024_sparse_lr1e-4_a0p05/cql_epoch0200_step0007000.pt`

Seçilen final model değerleri:

| Alan | Değer |
|---|---:|
| reward_variant | sparse |
| learning_rate | 1e-4 |
| cql_alpha | 0.05 |
| seed | 1024 |
| best_epoch | 200 |
| validation FQE | 15.670790 |
| WIS | 5.379920 |
| ESS | 7.824754 |
| CI/stability | Stage 2 artifact'te yok |
| support/coverage | Stage 2 artifact'te yok; ESS diagnostic olarak mevcut |

## Neden seçildi?

Seçilen model Stage 2 validation sonuçlarında en yüksek FQE'ye sahiptir: 15.670790. En yakın alternatif olan `shaped, lr=1e-3, alpha=0.1, seed=1024` modelinin validation FQE'si 15.661057'dir. Aradaki fark küçük olsa da seçilen modelin WIS=5.379920 ve ESS=7.824754 değerleri diagnostik olarak kabul edilebilir düzeydedir; yani seçim sadece yüksek FQE'ye bakılarak yapılmamış, WIS/ESS açısından belirgin bir diskalifiye sinyali görülmemiştir.

Seçilen model ayrıca düşük CQL alpha (0.05) kullanan sparse reward varyantıdır. Bu, final proposal'da alpha=0.05'in dahil edilme gerekçesiyle uyumludur: sepsis-benzeri offline CQL çalışmalarında daha düşük conservative penalty değerleri performanslı olabilmektedir.

## Alternatif adaylar neden elendi?

- `shaped, lr=1e-3, alpha=0.1, seed=1024`: FQE 15.661057 ile çok yakın ikinci adaydır. WIS=6.026481 ve ESS=8.182755 seçilen modelden biraz daha yüksek olsa da primary metric olan validation FQE daha düşüktür. Seçilen modelin ESS'i de yeterli düzeyde olduğu için bu aday final seçilmedi.
- `shaped, lr=1e-3, alpha=1.0, seed=1024`: FQE 15.633579 ile daha düşük kalmıştır. Ayrıca WIS=1.626519 ve ESS=5.172369, üst iki adaya göre daha zayıf diagnostic sinyal verir.
- `shaped, lr=1e-4, alpha=1.0, seed=1024`: ESS=9.243024 ve WIS=6.805624 ile diagnostic olarak güçlü görünür; ancak FQE 15.622208 ile seçilen modelden düşüktür. Primary objective FQE olduğu için elendi.
- `sparse, lr=1e-3, alpha=0.1, seed=1024`: Stage 1'de en iyi konfigürasyondu; fakat Stage 2 validation'da FQE 15.582195'e düştü ve seçilen Stage 2 adayının gerisinde kaldı.
- `sparse, lr=3e-4, alpha=1.0, seed=1024`: WIS=8.831675 yüksek görünse de FQE 15.320593 ile Stage 2 adayları arasında en düşük değere sahiptir.

## Test set seçimde kullanılmadı

Bu seçimde test set sonuçları kullanılmamıştır. Test split üzerinde herhangi bir model/config/checkpoint araması yapılmamalıdır. Test set sadece seçilen final checkpoint için, final raporlamaya yönelik tek seferlik değerlendirmede kullanılmalıdır.

## Final test evaluation komutu

Aşağıdaki komut sadece seçilmiş final CQL checkpoint'ini test split üzerinde değerlendirir. Tüm 24 run'ı veya tüm checkpoint'leri testte dolaşmaz.

```bash
uv run python scripts/evaluate_final_selected_policy.py \
  --checkpoint checkpoints/cql_sweep/cql_s1024_sparse_lr1e-4_a0p05/cql_epoch0200_step0007000.pt \
  --test-data data/replay/replay_test.parquet \
  --output runs/cql_sweep/final_test_evaluation.json \
  --bootstrap-resamples 1000
```

## Final test output dosyası

Final test çıktısı şu dosyaya yazılacaktır:

`runs/cql_sweep/final_test_evaluation.json`
