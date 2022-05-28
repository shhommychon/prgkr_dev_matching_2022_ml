# MindBigData ImageNet dataset ver 1.04
# http://www.mindbigdata.com/opendb/imagenet.html


# 압축파일 다운로드
from urllib import request
mindbigdata_url = "http://www.mindbigdata.com/opendb/MindBigData-Imagenet-IN.zip"
mindbigdata_zip = "MindBigData-Imagenet-IN.zip"
request.urlretrieve(mindbigdata_url, mindbigdata_zip)


# 압축파일 압축해제
import zipfile
with zipfile.ZipFile(mindbigdata_zip, 'r') as zip_file:
    zip_file.extractall("./")


# 압축파일 삭제
import os
os.remove(mindbigdata_zip)


# train:test == 0.8:0.2 로 분리
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
csv_list = glob("MindBigData-Imagenet/*.csv")
label_list = [ csv_file.split('_')[3] for csv_file in csv_list ]
train_label, test_label, train_csv, test_csv = train_test_split(label_list, csv_list, test_size=0.2, stratify=label_list, random_state=53)


# 파일 이동
if not os.path.exists("dataset0"):
    os.mkdir("dataset0")
if not os.path.exists("dataset0/train"):
    os.mkdir("dataset0/train")
if not os.path.exists("dataset0/test"):
    os.mkdir("dataset0/test")

class MyMindBigDataSuperclassifier:
    """569개의 원 class를 16개 상위 클래스로 재분류
    
    이 작업을 처음부터 다시 하려면 필요한 라이브러리 및 파일:
        robustness:                 https://github.com/MadryLab/robustness
        imagenet-mini dataset:      https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
        imagenet_class_index.json:  https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
        wordnet.is_a.txt:           https://github.com/innerlee/ImagenetSampling/blob/master/Imagenet/data/wordnet.is_a.txt
        words.txt:                  https://github.com/innerlee/ImagenetSampling/blob/master/Imagenet/data/words.txt
    
    ```
    from robustness.tools.imagenet_helpers import ImageNetHierarchy
    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid, class_ranges, label_map = in_hier.get_superclasses(n_classes, balanced=False)
    ```
    refer to https://robustness.readthedocs.io/en/latest/example_usage/custom_imagenet.html
    """
    def __init__(self):
        self.superclass_wnid = [
            'n00001740',
            'n00020827', 'n01503061', 'n01661818', 'n02075296', 'n02087122',
            'n02103406', 'n02159955', 'n03051540', 'n03100490', 'n03122748',
            'n03294048', 'n03563967', 'n03574816', 'n04170037', 'n04341686'
        ]
        
        self.child_to_parent_map = {
            # n00001740 - entity
            "n00001740": "n00001740", # entity

            # n00020827 - matter
            "n07583066": "n00020827", # guacamole
            "n07615774": "n00020827", # ice lolly, lolly, lollipop, popsicle
            "n07693725": "n00020827", # bagel, beigel
            "n07695742": "n00020827", # pretzel
            "n07697100": "n00020827", # hamburger, beefburger, burger
            "n07697313": "n00020827", # cheeseburger
            "n07697537": "n00020827", # hotdog, hot dog, red hot
            "n07714571": "n00020827", # head cabbage
            "n07718472": "n00020827", # cucumber, cuke
            "n07718747": "n00020827", # artichoke, globe artichoke
            "n07720875": "n00020827", # bell pepper
            "n07734744": "n00020827", # mushroom
            "n07745940": "n00020827", # strawberry
            "n07747607": "n00020827", # orange
            "n07749582": "n00020827", # lemon
            "n07753113": "n00020827", # fig
            "n07753275": "n00020827", # pineapple, ananas
            "n07753592": "n00020827", # banana
            "n07768694": "n00020827", # pomegranate
            "n07873807": "n00020827", # pizza, pizza pie
            "n07880968": "n00020827", # burrito

            # n01503061 - bird
            "n01503061": "n01503061", # bird
            "n01514668": "n01503061", # cock
            "n01514859": "n01503061", # hen
            "n01518878": "n01503061", # ostrich, Struthio camelus
            "n01530575": "n01503061", # brambling, Fringilla montifringilla
            "n01531178": "n01503061", # goldfinch, Carduelis carduelis
            "n01532829": "n01503061", # house finch, linnet, Carpodacus mexicanus
            "n01534433": "n01503061", # junco, snowbird
            "n01537544": "n01503061", # indigo bunting, indigo finch, indigo bird, Passerina cyanea
            "n01558993": "n01503061", # robin, American robin, Turdus migratorius
            "n01560419": "n01503061", # bulbul
            "n01580077": "n01503061", # jay
            "n01582220": "n01503061", # magpie
            "n01592084": "n01503061", # chickadee
            "n01601694": "n01503061", # water ouzel, dipper
            "n01608432": "n01503061", # kite
            "n01614925": "n01503061", # bald eagle, American eagle, Haliaeetus leucocephalus
            "n01616318": "n01503061", # vulture
            "n01622779": "n01503061", # great grey owl, great gray owl, Strix nebulosa
            "n01817953": "n01503061", # African grey, African gray, Psittacus erithacus
            "n01818515": "n01503061", # macaw
            "n01819313": "n01503061", # sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita
            "n01820546": "n01503061", # lorikeet
            "n01824575": "n01503061", # coucal
            "n01828970": "n01503061", # bee eater
            "n01829413": "n01503061", # hornbill
            "n01833805": "n01503061", # hummingbird
            "n01843065": "n01503061", # jacamar
            "n01843383": "n01503061", # toucan
            "n01847000": "n01503061", # drake
            "n01855032": "n01503061", # red-breasted merganser, Mergus serrator
            "n01855672": "n01503061", # goose
            "n01860187": "n01503061", # black swan, Cygnus atratus
            "n02002556": "n01503061", # white stork, Ciconia ciconia
            "n02002724": "n01503061", # black stork, Ciconia nigra
            "n02006656": "n01503061", # spoonbill
            "n02007558": "n01503061", # flamingo
            "n02009229": "n01503061", # little blue heron, Egretta caerulea
            "n02009912": "n01503061", # American egret, great white heron, Egretta albus
            "n02011460": "n01503061", # bittern
            "n02012849": "n01503061", # crane
            "n02013706": "n01503061", # limpkin, Aramus pictus
            "n02017213": "n01503061", # European gallinule, Porphyrio porphyrio
            "n02018207": "n01503061", # American coot, marsh hen, mud hen, water hen, Fulica americana
            "n02018795": "n01503061", # bustard
            "n02025239": "n01503061", # ruddy turnstone, Arenaria interpres
            "n02027492": "n01503061", # red-backed sandpiper, dunlin, Erolia alpina
            "n02028035": "n01503061", # redshank, Tringa totanus
            "n02033041": "n01503061", # dowitcher
            "n02037110": "n01503061", # oystercatcher, oyster catcher
            "n02051845": "n01503061", # pelican
            "n02056570": "n01503061", # king penguin, Aptenodytes patagonica
            "n02058221": "n01503061", # albatross, mollymawk

            # n01661818 - diapsid, diapsid reptile
            "n01674464": "n01661818", # lizard
            "n01675722": "n01661818", # banded gecko
            "n01677366": "n01661818", # common iguana, iguana, Iguana iguana
            "n01682714": "n01661818", # American chameleon, anole, Anolis carolinensis
            "n01685808": "n01661818", # whiptail, whiptail lizard
            "n01687978": "n01661818", # agama
            "n01688243": "n01661818", # frilled lizard, Chlamydosaurus kingi
            "n01689811": "n01661818", # alligator lizard
            "n01692333": "n01661818", # Gila monster, Heloderma suspectum
            "n01693334": "n01661818", # green lizard, Lacerta viridis
            "n01694178": "n01661818", # African chameleon, Chamaeleo chamaeleon
            "n01695060": "n01661818", # Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis
            "n01726692": "n01661818", # snake, serpent, ophidian
            "n01728572": "n01661818", # thunder snake, worm snake, Carphophis amoenus
            "n01728920": "n01661818", # ringneck snake, ring-necked snake, ring snake
            "n01729322": "n01661818", # hognose snake, puff adder, sand viper
            "n01729977": "n01661818", # green snake, grass snake
            "n01734418": "n01661818", # king snake, kingsnake
            "n01735189": "n01661818", # garter snake, grass snake
            "n01737021": "n01661818", # water snake
            "n01739381": "n01661818", # vine snake
            "n01740131": "n01661818", # night snake, Hypsiglena torquata
            "n01742172": "n01661818", # boa constrictor, Constrictor constrictor
            "n01744401": "n01661818", # rock python, rock snake, Python sebae
            "n01748264": "n01661818", # Indian cobra, Naja naja
            "n01749939": "n01661818", # green mamba
            "n01751748": "n01661818", # sea snake
            "n01753488": "n01661818", # horned viper, cerastes, sand viper, horned asp, Cerastes cornutus
            "n01755581": "n01661818", # diamondback, diamondback rattlesnake, Crotalus adamanteus
            "n01756291": "n01661818", # sidewinder, horned rattlesnake, Crotalus cerastes

            # n02075296 - carnivore
            "n02118333": "n02075296", # fox
            "n02119022": "n02075296", # red fox, Vulpes vulpes
            "n02119789": "n02075296", # kit fox, Vulpes macrotis
            "n02120079": "n02075296", # Arctic fox, white fox, Alopex lagopus
            "n02120505": "n02075296", # grey fox, gray fox, Urocyon cinereoargenteus
            "n02129165": "n02075296", # lion, king of beasts, Panthera leo
            "n02129604": "n02075296", # tiger, Panthera tigris
            "n02131653": "n02075296", # bear
            "n02132136": "n02075296", # brown bear, bruin, Ursus arctos
            "n02133161": "n02075296", # American black bear, black bear, Ursus americanus, Euarctos americanus
            "n02134084": "n02075296", # ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus
            "n02134418": "n02075296", # sloth bear, Melursus ursinus, Ursus ursinus
            "n02444819": "n02075296", # otter
            "n02445715": "n02075296", # skunk, polecat, wood pussy
            "n02509815": "n02075296", # lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
            "n02510455": "n02075296", # giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca

            # n02087122 - hunting dog
            "n02087394": "n02087122", # Rhodesian ridgeback
            "n02088094": "n02087122", # Afghan hound, Afghan
            "n02088238": "n02087122", # basset, basset hound
            "n02088364": "n02087122", # beagle
            "n02088466": "n02087122", # bloodhound, sleuthhound
            "n02088632": "n02087122", # bluetick
            "n02089078": "n02087122", # black-and-tan coonhound
            "n02089867": "n02087122", # Walker hound, Walker foxhound
            "n02089973": "n02087122", # English foxhound
            "n02090379": "n02087122", # redbone
            "n02090622": "n02087122", # borzoi, Russian wolfhound
            "n02090721": "n02087122", # Irish wolfhound
            "n02091244": "n02087122", # Ibizan hound, Ibizan Podenco
            "n02091467": "n02087122", # Norwegian elkhound, elkhound
            "n02091635": "n02087122", # otterhound, otter hound
            "n02091831": "n02087122", # Saluki, gazelle hound
            "n02092002": "n02087122", # Scottish deerhound, deerhound
            "n02092339": "n02087122", # Weimaraner
            "n02093256": "n02087122", # Staffordshire bullterrier, Staffordshire bull terrier
            "n02093428": "n02087122", # American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier
            "n02093647": "n02087122", # Bedlington terrier
            "n02093754": "n02087122", # Border terrier
            "n02093859": "n02087122", # Kerry blue terrier
            "n02093991": "n02087122", # Irish terrier
            "n02094114": "n02087122", # Norfolk terrier
            "n02094258": "n02087122", # Norwich terrier
            "n02094433": "n02087122", # Yorkshire terrier
            "n02095314": "n02087122", # wire-haired fox terrier
            "n02095570": "n02087122", # Lakeland terrier
            "n02095889": "n02087122", # Sealyham terrier, Sealyham
            "n02096051": "n02087122", # Airedale, Airedale terrier
            "n02096177": "n02087122", # cairn, cairn terrier
            "n02096294": "n02087122", # Australian terrier
            "n02096437": "n02087122", # Dandie Dinmont, Dandie Dinmont terrier
            "n02096585": "n02087122", # Boston bull, Boston terrier
            "n02097047": "n02087122", # miniature schnauzer
            "n02097130": "n02087122", # giant schnauzer
            "n02097209": "n02087122", # standard schnauzer
            "n02097298": "n02087122", # Scotch terrier, Scottish terrier, Scottie
            "n02097474": "n02087122", # Tibetan terrier, chrysanthemum dog
            "n02097658": "n02087122", # silky terrier, Sydney silky
            "n02098105": "n02087122", # soft-coated wheaten terrier
            "n02098286": "n02087122", # West Highland white terrier
            "n02098413": "n02087122", # Lhasa, Lhasa apso
            "n02099267": "n02087122", # flat-coated retriever
            "n02099429": "n02087122", # curly-coated retriever
            "n02099601": "n02087122", # golden retriever
            "n02099712": "n02087122", # Labrador retriever
            "n02099849": "n02087122", # Chesapeake Bay retriever
            "n02100236": "n02087122", # German short-haired pointer
            "n02100583": "n02087122", # vizsla, Hungarian pointer
            "n02100735": "n02087122", # English setter
            "n02100877": "n02087122", # Irish setter, red setter
            "n02101006": "n02087122", # Gordon setter
            "n02101388": "n02087122", # Brittany spaniel
            "n02101556": "n02087122", # clumber, clumber spaniel
            "n02102040": "n02087122", # English springer, English springer spaniel
            "n02102177": "n02087122", # Welsh springer spaniel
            "n02102318": "n02087122", # cocker spaniel, English cocker spaniel, cocker
            "n02102480": "n02087122", # Sussex spaniel
            "n02102973": "n02087122", # Irish water spaniel

            # n02103406 - working dog
            "n02104029": "n02103406", # kuvasz
            "n02104365": "n02103406", # schipperke
            "n02105056": "n02103406", # groenendael
            "n02105162": "n02103406", # malinois
            "n02105251": "n02103406", # briard
            "n02105412": "n02103406", # kelpie
            "n02105505": "n02103406", # komondor
            "n02105641": "n02103406", # Old English sheepdog, bobtail
            "n02105855": "n02103406", # Shetland sheepdog, Shetland sheep dog, Shetland
            "n02106030": "n02103406", # collie
            "n02106166": "n02103406", # Border collie
            "n02106382": "n02103406", # Bouvier des Flandres, Bouviers des Flandres
            "n02106550": "n02103406", # Rottweiler
            "n02106662": "n02103406", # German shepherd, German shepherd dog, German police dog, alsatian
            "n02107142": "n02103406", # Doberman, Doberman pinscher
            "n02107312": "n02103406", # miniature pinscher
            "n02107574": "n02103406", # Greater Swiss Mountain dog
            "n02107683": "n02103406", # Bernese mountain dog
            "n02107908": "n02103406", # Appenzeller
            "n02108000": "n02103406", # EntleBucher
            "n02108089": "n02103406", # boxer
            "n02108422": "n02103406", # bull mastiff
            "n02108551": "n02103406", # Tibetan mastiff
            "n02108915": "n02103406", # French bulldog
            "n02109047": "n02103406", # Great Dane
            "n02109525": "n02103406", # Saint Bernard, St Bernard
            "n02109961": "n02103406", # Eskimo dog, husky
            "n02110063": "n02103406", # malamute, malemute, Alaskan malamute
            "n02110185": "n02103406", # Siberian husky
            "n02110627": "n02103406", # affenpinscher, monkey pinscher, monkey dog

            # n02159955 - insect
            "n02165456": "n02159955", # ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle
            "n02206856": "n02159955", # bee
            "n02219486": "n02159955", # ant, emmet, pismire
            "n02268443": "n02159955", # dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk
            "n02274259": "n02159955", # butterfly
            "n02276258": "n02159955", # admiral
            "n02277742": "n02159955", # ringlet, ringlet butterfly
            "n02279972": "n02159955", # monarch, monarch butterfly, milkweed butterfly, Danaus plexippus
            "n02280649": "n02159955", # cabbage butterfly
            "n02281406": "n02159955", # sulphur butterfly, sulfur butterfly
            "n02281787": "n02159955", # lycaenid, lycaenid butterfly

            # n03051540 - clothing, article of clothing, vesture, wear, wearable, habiliment
            "n02807133": "n03051540", # bathing cap, swimming cap
            "n02883205": "n03051540", # bow tie, bow-tie, bowtie
            "n02892767": "n03051540", # brassiere, bra, bandeau
            "n03124170": "n03051540", # cowboy hat, ten-gallon hat
            "n03127747": "n03051540", # crash helmet
            "n03188531": "n03051540", # diaper, nappy, napkin
            "n03379051": "n03051540", # football helmet
            "n03513137": "n03051540", # helmet
            "n03710721": "n03051540", # maillot, tank suit
            "n03770439": "n03051540", # miniskirt, mini
            "n04209133": "n03051540", # shower cap
            "n04259630": "n03051540", # sombrero
            "n04371430": "n03051540", # swimming trunks, bathing trunks
            "n04591157": "n03051540", # Windsor tie

            # n03100490 - conveyance, transport
            "n02687172": "n03100490", # aircraft carrier, carrier, flattop, attack aircraft carrier
            "n02690373": "n03100490", # airliner
            "n02691156": "n03100490", # airplane, aeroplane, plane
            "n02917067": "n03100490", # bullet train, bullet
            "n02924116": "n03100490", # bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle
            "n02951358": "n03100490", # canoe
            "n02981792": "n03100490", # catamaran
            "n03095699": "n03100490", # container ship, containership, container vessel
            "n03344393": "n03100490", # fireboat
            "n03447447": "n03100490", # gondola
            "n03662601": "n03100490", # lifeboat
            "n03673027": "n03100490", # liner, ocean liner
            "n03769881": "n03100490", # minibus
            "n03947888": "n03100490", # pirate, pirate ship
            "n04146614": "n03100490", # school bus
            "n04147183": "n03100490", # schooner
            "n04273569": "n03100490", # speedboat
            "n04336792": "n03100490", # stretcher
            "n04347754": "n03100490", # submarine, pigboat, sub, U-boat
            "n04468005": "n03100490", # train, railroad train
            "n04483307": "n03100490", # trimaran
            "n04487081": "n03100490", # trolleybus, trolley coach, trackless trolley
            "n04530566": "n03100490", # vessel, watercraft
            "n04606251": "n03100490", # wreck
            "n04612504": "n03100490", # yawl

            # n03122748 - covering
            "n02786058": "n03122748", # Band Aid
            "n02840245": "n03122748", # binder, ring-binder

            # n03294048 - equipment
            "n02777292": "n03294048", # balance beam, beam
            "n02799071": "n03294048", # baseball
            "n02802426": "n03294048", # basketball
            "n02979186": "n03294048", # cassette player
            "n03085013": "n03294048", # computer keyboard, keypad
            "n03134739": "n03294048", # croquet ball
            "n03255030": "n03294048", # dumbbell
            "n03445777": "n03294048", # golf ball
            "n03445924": "n03294048", # golfcart, golf cart
            "n03535780": "n03294048", # horizontal bar, high bar
            "n03782006": "n03294048", # monitor
            "n03942813": "n03294048", # ping-pong ball
            "n03982430": "n03294048", # pool table, billiard table, snooker table
            "n04004767": "n03294048", # printer
            "n04023962": "n03294048", # punching bag, punch bag, punching ball, punchball
            "n04118538": "n03294048", # rugby ball
            "n04254680": "n03294048", # soccer ball
            "n04392985": "n03294048", # tape player
            "n04409515": "n03294048", # tennis ball
            "n04540053": "n03294048", # volleyball

            # n03563967 - implement
            "n02764044": "n03563967", # ax, axe
            "n02951585": "n03563967", # can opener, tin opener
            "n03109150": "n03563967", # corkscrew, bottle screw
            "n03141823": "n03563967", # crutch
            "n03400231": "n03563967", # frying pan, frypan, skillet
            "n03481172": "n03563967", # hammer
            "n03498962": "n03563967", # hatchet
            "n03908714": "n03563967", # pencil sharpener
            "n04039381": "n03563967", # racket, racquet
            "n04116512": "n03563967", # rubber eraser, rubber, pencil eraser
            "n04154565": "n03563967", # screwdriver
            "n04270147": "n03563967", # spatula

            # n03574816 - instrument
            "n02879718": "n03574816", # bow
            "n03196217": "n03574816", # digital clock
            "n04118776": "n03574816", # rule, ruler
            "n04317175": "n03574816", # stethoscope
            "n04356056": "n03574816", # sunglasses, dark glasses, shades
            "n04376876": "n03574816", # syringe

            # n04170037 - self-propelled vehicle
            "n02701002": "n04170037", # ambulance
            "n02814533": "n04170037", # beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
            "n02930766": "n04170037", # cab, hack, taxi, taxicab
            "n02958343": "n04170037", # car, auto, automobile, machine, motorcar
            "n03100240": "n04170037", # convertible
            "n03594945": "n04170037", # jeep, landrover
            "n03670208": "n04170037", # limousine, limo
            "n03770679": "n04170037", # minivan
            "n03777568": "n04170037", # Model T
            "n03785016": "n04170037", # moped
            "n03790512": "n04170037", # motorcycle, bike
            "n04037443": "n04170037", # racer, race car, racing car
            "n04252077": "n04170037", # snowmobile
            "n04252225": "n04170037", # snowplow, snowplough
            "n04285008": "n04170037", # sports car, sport car

            # n04341686 - structure, construction
            "n02821627": "n04341686", # bedroom, sleeping room, sleeping accommodation, chamber, bedchamber
            "n02839592": "n04341686", # billiard room, billiard saloon, billiard parlor, billiard parlour, billiard hall
            "n03038685": "n04341686", # classroom, schoolroom
            "n03120778": "n04341686", # court, courtroom
            "n03416640": "n04341686", # garage, service department
            "n03529860": "n04341686", # home theater, home theatre
            "n03619890": "n04341686", # kitchen
            "n03841666": "n04341686", # office, business office
            "n03961711": "n04341686", # plate rack
            "n04065464": "n04341686", # recreation room, rec room
            "n04081281": "n04341686", # restaurant, eating house, eating place, eatery
        }
        
    def superclass(self):
        return self.superclass_wnid

    def map(self, child):
        return self.child_to_parent_map[child]

reclassify = MyMindBigDataSuperclassifier()

from shutil import copy

for l, c in zip(train_label, train_csv):
    try:
        l = reclassify.map(l)
    except:
        l = "n00001740"
    if not os.path.exists(f"dataset0/train/{l}"):
        os.mkdir(f"dataset0/train/{l}")
    
    df = pd.read_csv(c, header=None)
    df = df.T
    df.columns = df.iloc[0, :]
    df = df.iloc[1:, :].reset_index(drop=True)
    df.to_csv(f"dataset0/train/{l}/{c.split('/')[-1]}", index=False)

for l, c in zip(test_label, test_csv):
    try:
        l = reclassify.map(l)
    except:
        l = "n00001740"
    if not os.path.exists(f"dataset0/test/{l}"):
        os.mkdir(f"dataset0/test/{l}")
    
    df = pd.read_csv(c, header=None)
    df = df.T
    df.columns = df.iloc[0, :]
    df = df.iloc[1:, :].reset_index(drop=True)
    df.to_csv(f"dataset0/test/{l}/{c.split('/')[-1]}", index=False)

for folder in os.listdir("dataset0/train"):
    print(folder)
    print('\t', len(os.listdir(f"dataset0/train/{folder}")))
    print('\t', len(os.listdir(f"dataset0/test/{folder}")))


# 불필요 데이터 삭제
from shutil import rmtree
os.remove("WordReport-v1.04.txt")
rmtree("MindBigData-Imagenet")
rmtree("dataset0/train/n00001740")
rmtree("dataset0/test/n00001740")
# ※참고: 원 시험에서도 16개 클래스라고 설명했지만 15개 클래스의 데이터를 제공하였다
