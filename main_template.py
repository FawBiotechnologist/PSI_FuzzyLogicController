#
# Podstawy Sztucznej Inteligencji, IIS 2020
# Autor: Tomasz Jaworski
# Opis: Szablon kodu do stabilizacji odwróconego wahadła (patyka) w pozycji pionowej podczas ruchu wózka.
#

import gym  # Instalacja: https://github.com/openai/gym
import time
from helper import HumanControl, Keys, CartForce
import matplotlib.pyplot as plt

import numpy as np
import skfuzzy as fuzz

#
# przygotowanie środowiska
#
control = HumanControl()
env = gym.make('gym_PSI:CartPole-v2')
env.reset()
env.render()
noFuzzy = False


def generateMembershipFunction(variable, range):
    left = fuzz.trimf(variable, [-range, -range, 0])
    mid = fuzz.trimf(variable, [-range / 100, 0, range / 100])
    right = fuzz.trimf(variable, [0, range, range])
    return left, mid, right


# def generateMembershipFunction(variable, range):
#     left = fuzz.trimf(variable, [-range, -range, 0])
#     mid = fuzz.trimf(variable, [-0.1, 0, 0.1])
#     right = fuzz.trimf(variable, [0, range, range])
#     return left, mid, right


# def generateMembershipFunction(variable, range):
#     left = fuzz.trimf(variable, [-range, -range / 10, -range / 100])
#     mid = fuzz.trimf(variable, [-range / 100, 0, range / 100])
#     right = fuzz.trimf(variable, [range / 100, range / 10, range])
#     return left, mid, right


# def generateForceMembershipFunction(variable, range):
#     left = fuzz.trimf(variable, [-range, -range / 2, 0])
#     mid = fuzz.trimf(variable, [-range / 1000, 0, range / 1000])
#     right = fuzz.trimf(variable, [0, range / 2, range])
#     return left, mid, right

# def generateForceMembershipFunction(variable, range):
#     left = fuzz.trimf(variable, [-range, -range / 2, 0])
#     mid = fuzz.trimf(variable, [-range / 4, 0, range / 4])
#     right = fuzz.trimf(variable, [0, range / 2, range])
#     return left, mid, right

def generateForceMembershipFunction(variable, range):
    left = fuzz.trapmf(variable, [-range * 2, -range / 2, -range / 4, 0])
    mid = fuzz.trapmf(variable, [-range, -range / 4, range / 4, range])
    right = fuzz.trapmf(variable, [0, range / 4, range/2, range * 2])
    return left, mid, right


def on_key_press(key: int, mod: int):
    global control
    force = 10
    if key == Keys.LEFT:
        control.UserForce = force * CartForce.UNIT_LEFT  # krok w lewo
    if key == Keys.RIGHT:
        control.UserForce = force * CartForce.UNIT_RIGHT  # krok w prawo
    if key == Keys.P:  # pauza
        control.WantPause = True
    if key == Keys.R:  # restart
        control.WantReset = True
    if key == Keys.ESCAPE or key == Keys.Q:  # wyjście
        control.WantExit = True


env.unwrapped.viewer.window.on_key_press = on_key_press

#########################################################
# KOD INICJUJĄCY - do wypełnienia
#########################################################

"""

1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
3. Wyświetl je, w celach diagnostycznych.

Przykład wyświetlania:

fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))

ax0.plot(x_variable, variable_left, 'b', linewidth=1.5, label='Left')
ax0.plot(x_variable, variable_zero, 'g', linewidth=1.5, label='Zero')
ax0.plot(x_variable, variable_right, 'r', linewidth=1.5, label='Right')
ax0.set_title('Angle')
ax0.legend()


plt.tight_layout()
plt.show()
"""
fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))
force_variable = np.arange(-20, 20, 0.01)
force_variable_left, force_variable_zero, force_variable_right = generateForceMembershipFunction(force_variable, 20)
angle_variable = np.arange(-1, 1, 0.01)
angle_variable_left, angle_variable_zero, angle_variable_right = generateMembershipFunction(angle_variable, 1)
cart_velocity_variable = np.arange(-2, 2, 0.01)
cart_velocity_variable_left, cart_velocity_variable_zero, cart_velocity_variable_right = generateMembershipFunction(
    cart_velocity_variable, 2)
cart_pos_variable = np.arange(-4, 4, 0.01)
cart_pos_variable_left, cart_pos_variable_zero, cart_pos_variable_right = generateMembershipFunction(
    cart_pos_variable, 4)
ax0.plot(angle_variable, angle_variable_left, 'b', linewidth=1.5, label='Left')
ax0.plot(angle_variable, angle_variable_zero, 'g', linewidth=1.5, label='Zero')
ax0.plot(angle_variable, angle_variable_right, 'r', linewidth=1.5, label='Right')
ax0.set_title('Angle')
ax0.legend()

plt.tight_layout()
# plt.show()
#########################################################
# KONIEC KODU INICJUJĄCEGO
#########################################################


#
# Główna pętla symulacji
#
while not control.WantExit:

    #
    # Wstrzymywanie symulacji:
    # Pierwsze wciśnięcie klawisza 'p' wstrzymuje; drugie wciśnięcie 'p' wznawia symulację.
    #
    if control.WantPause:
        control.WantPause = False
        while not control.WantPause:
            time.sleep(0.1)
            env.render()
        control.WantPause = False

    #
    # Czy użytkownik chce zresetować symulację?
    if control.WantReset:
        control.WantReset = False
        env.reset()

    ###################################################
    # ALGORYTM REGULACJI - do wypełnienia
    ##################################################

    """
    Opis wektora stanu (env.state)
        cart_position   -   Położenie wózka w osi X. Zakres: -2.5 do 2.5. Ppowyżej tych granic wózka znika z pola widzenia.
        cart_velocity   -   Prędkość wózka. Zakres +- Inf, jednak wartości powyżej +-2.0 generują zbyt szybki ruch.
        pole_angle      -   Pozycja kątowa patyka, a<0 to odchylenie w lewo, a>0 odchylenie w prawo. Pozycja kątowa ma
                            charakter bezwzględny - do pozycji wliczane są obroty patyka.
                            Ze względów intuicyjnych zaleca się konwersję na stopnie (+-180).
        tip_velocity    -   Prędkość wierzchołka patyka. Zakres +- Inf. a<0 to ruch przeciwny do wskazówek zegara,
                            podczas gdy a>0 to ruch zgodny z ruchem wskazówek zegara.
                            
    Opis zadajnika akcji (fuzzy_response):
        Jest to wartość siły przykładana w każdej chwili czasowej symulacji, wyrażona w Newtonach.
        Zakładany krok czasowy symulacji to env.tau (20 ms).
        Przyłożenie i utrzymanie stałej siły do wózka spowoduje, że ten będzie przyspieszał do nieskończoności,
        ruchem jednostajnym.
    """

    cart_position, cart_velocity, pole_angle, tip_velocity = env.state  # Wartości zmierzone

    """
    
    1. Przeprowadź etap rozmywania, w którym dla wartości zmierzonych wyznaczone zostaną ich przynależności do poszczególnych
       zmiennych lingwistycznych. Jedno fizyczne wejście (źródło wartości zmierzonych, np. położenie wózka) posiada własną
       zmienną lingwistyczną.
       
       Sprawdź funkcję interp_membership
       """
    poleLeft, poleZero, poleRight = fuzz.interp_membership(angle_variable, angle_variable_left, pole_angle), \
                                    fuzz.interp_membership(angle_variable, angle_variable_zero, pole_angle), \
                                    fuzz.interp_membership(angle_variable, angle_variable_right, pole_angle)

    cartMovingLeft, cartIdle, cartMovingRight = fuzz.interp_membership(cart_velocity_variable,
                                                                       cart_velocity_variable_left, cart_velocity), \
                                                fuzz.interp_membership(cart_velocity_variable,
                                                                       cart_velocity_variable_zero, cart_velocity), \
                                                fuzz.interp_membership(cart_velocity_variable,
                                                                       cart_velocity_variable_right, cart_velocity)
    cartPosLeft, cartPosIdle, cartPosRight = fuzz.interp_membership(cart_pos_variable,
                                                                    cart_pos_variable_left, cart_position), \
                                             fuzz.interp_membership(cart_pos_variable,
                                                                    cart_pos_variable_zero, cart_position), \
                                             fuzz.interp_membership(cart_pos_variable,
                                                                    cart_pos_variable_right, cart_position)

    """
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.
       Przykład reguły:
       JEŻELI kąt patyka jest zerowy ORAZ prędkość wózka jest zerowa TO moc chwilowa jest zerowa
       JEŻELI kąt patyka jest lekko ujemny ORAZ prędkość wózka jest zerowa TO moc chwilowa jest lekko ujemna
       JEŻELI kąt patyka jest średnio ujemny ORAZ prędkość wózka jest lekko ujemna TO moc chwilowa jest średnio ujemna
       JEŻELI kąt patyka jest szybko rosnący w kierunku ujemnym TO moc chwilowa jest mocno ujemna
       .....
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.
       """


    def OR(a, b):
        return np.fmax(a, b)


    def AND(a, b):
        return np.fmin(a, b)


    """
    
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    """
    # L = OR(cartMovingRight, AND(poleLeft, cartPosRight))
    # R = OR(cartMovingLeft, AND(poleRight, cartPosLeft))
    # I = AND(cartIdle, AND(poleZero, cartPosIdle))

    # L = OR(poleLeft, OR(AND(poleZero, cartPosRight), AND(poleZero, cartMovingRight)))
    # R = OR(poleRight, OR(AND(poleZero, cartPosLeft), AND(poleZero, cartMovingLeft)))
    # I = AND(cartIdle, AND(poleZero, cartPosIdle))

    # L = OR(poleLeft, AND(cartPosRight, cartMovingRight))
    # R = OR(poleRight, AND(cartPosLeft, cartMovingLeft))
    # I = AND(AND(cartPosIdle, poleZero), cartIdle)

    # L = OR(poleLeft, AND(cartPosRight, poleZero))
    # R = OR(poleRight, AND(cartPosLeft, poleZero))
    # I = AND(poleZero, AND(cartPosIdle,cartIdle))

    # L = poleLeft
    # R = poleRight
    # I = poleZero

    L = OR(poleLeft, AND(OR(poleLeft, poleZero), AND(cartMovingRight, cartPosRight)))
    R = OR(poleRight, AND(OR(poleRight, poleZero), AND(cartMovingLeft, cartPosLeft)))
    I = OR(AND(poleZero, cartPosIdle), AND(cartIdle, poleZero))
    # poleLeft = 10 cart pos = 2
    # BEST?
    # L = OR(OR(poleLeft, poleZero), AND(cartMovingRight, cartPosRight))
    # R = OR(OR(poleRight, poleZero), AND(cartMovingLeft, cartPosLeft))
    # I = OR(AND(poleZero, AND(cartIdle, cartPosIdle)),
    #        OR(AND(cartMovingLeft, AND(cartPosRight, poleLeft)), AND(cartMovingRight, AND(cartPosLeft, poleRight))))
    # L = OR(poleLeft, AND(OR(poleLeft, poleZero), OR(cartMovingRight, cartPosRight)))
    # R = OR(poleRight, AND(OR(poleRight, poleZero), OR(cartMovingLeft, cartPosLeft)))
    # I = OR(AND(poleZero, AND(cartIdle, cartPosIdle)),
    #        OR(AND(cartMovingLeft, AND(cartPosRight, poleLeft)), AND(cartMovingRight, AND(cartPosLeft, poleRight))))
    # L = AND(poleLeft, OR(cartMovingRight, cartPosRight))
    # R = AND(poleRight, OR(cartMovingLeft, cartPosLeft))
    # I = poleZero

    """
    
    4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
       Operatorem wnioskowania jest min().
       Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
       to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
       
       W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
       Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.
       Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. 
       """
    left = np.fmin(force_variable_left, L)
    right = np.fmin(force_variable_right, R)
    idle = np.fmin(force_variable_zero, I)
    """
    5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
    """
    aggregatedActivations = OR(idle, OR(left, right))
    """
    6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).
        """
    fuzzy_response = fuzz.defuzz(force_variable, aggregatedActivations, 'centroid')
    # print("Fuzzy Response = :" + str(fuzzy_response))
    """

    7. Czym będzie wyjściowa wartość skalarna?
    
    """


    def noFuzzy():
        if tip_velocity < 0:
            # fuzzy_response = tip_velocity * abs(cart_velocity)
            fuzzy_response = CartForce.UNIT_LEFT
            if pole_angle <= -0.01:
                fuzzy_response = CartForce.UNIT_LEFT * 10  # do zmiennej fuzzy_response zapisz wartość siły, jaką chcesz przyłożyć do wózka.
        elif tip_velocity > 0:
            # fuzzy_response = tip_velocity * abs(cart_velocity)
            fuzzy_response = CartForce.UNIT_RIGHT
            if pole_angle >= 0.01:
                fuzzy_response = CartForce.UNIT_RIGHT * 10
        else:
            fuzzy_response = CartForce.IDLE_FORCE
        return fuzzy_response


    if noFuzzy is True:
        fuzzy_response = noFuzzy()
    #
    # KONIEC algorytmu regulacji
    #########################
    # Jeżeli użytkownik chce przesunąć wózek, to jego polecenie ma wyższy priorytet
    if control.UserForce is not None:
        applied_force = control.UserForce
        control.UserForce = None
    else:
        applied_force = fuzzy_response

    #
    # Wyświetl stan środowiska oraz wartość odpowiedzi regulatora na ten stan.
    print(
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()
    # time.sleep(1)

#
# Zostaw ten patyk!
env.close()
